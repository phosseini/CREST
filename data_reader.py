import os
import copy
import re
import spacy
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from nltk.corpus import framenet as fn
from nltk.corpus import verbnet as vn
from nltk.corpus import wordnet as wn


class CausalDataReader:
    def __init__(self):
        self.dir_path = "data/causal/"
        self.train_path = 'data/snorkel/train.csv'
        self.gold_causal_path = 'data/causal/gold_causal.csv'
        # direction = 0 -> (e1, e2), otherwise, (e2, e1)
        # split -> 0: train, 1: dev, test: 2
        self.scheme_columns = ['id', 'arg1', 'arg2', 'text', 'direction', 'label', 'source', 'ann_file', 'split']
        self.connective_columns = ['word', 'count', 'type', 'temporal', 'flag']
        self.arg1_tag = ["<@rg1>", "</@rg1>"]
        self.arg2_tag = ["<@rg2>", "</@rg2>"]
        self.signal_tag = ["<S!G>", "</S!G>"]
        # loading spaCy's english model
        self.nlp = spacy.load("en_core_web_sm")
        self.semeval_2007_4_code = 1
        self.semeval_2010_8_code = 2
        self.event_causality_code = 3
        self.causal_timebank_code = 4
        self.storyline_code = 5
        self.caters_code = 6
        self.because_code = 7

    def load_snorkel_data(self, n_train=11000, n_sample=210):
        """
        loading train (no label), dev, and test sets
        :param n_train: number of training samples
        :param n_sample: number of positive/negative samples.
        :return:
        """

        if os.path.isfile(self.train_path):
            # ========================================
            # ********** loading train set ***********
            # ========================================
            train = pd.read_csv(self.train_path)
            train = train.reset_index(drop=True)
            train = train.reindex(np.random.RandomState(seed=42).permutation(train.index))
            train = train.sample(n=n_train, random_state=123)
            train = train.reset_index(drop=True)

            # ========================================
            # ****** creating DEV and TEST sets ******
            # ========================================
            # [0] reading all the gold causal samples
            if os.path.exists(self.gold_causal_path):
                data = pd.read_csv(self.gold_causal_path)
            else:
                data = self.read_all()

            # [1] making sure we have a balanced set of causal and non-causal samples
            data = data.reset_index(drop=True)
            neg_data = data.loc[data.label == 0]
            pos_data = data.loc[data.label == 1]

            # [2] permutation of data
            neg_data = neg_data.reindex(np.random.RandomState(seed=42).permutation(neg_data.index))
            pos_data = pos_data.reindex(np.random.RandomState(seed=42).permutation(pos_data.index))

            # [3] getting samples of causal and non-causal
            neg_data = neg_data.sample(n=n_sample, random_state=42)
            # [3.1] getting same # of samples from different sources
            pos_data = pos_data.groupby('source').apply(lambda x: x.sample(n=30, random_state=42))

            # [4] storing all samples in one data frame
            df = copy.deepcopy(neg_data)
            df = df.append(pos_data)
            df = df.reindex(np.random.RandomState(seed=42).permutation(df.index))
            df = df.reset_index(drop=True)

            # [5] creating final dev and test sets
            dev = df.groupby('label').apply(lambda x: x.sample(frac=0.5, random_state=42))
            test = df.drop(dev.index.levels[1])

            dev = copy.deepcopy(dev[self.scheme_columns])
            test = copy.deepcopy(test[self.scheme_columns])

            return (dev, dev['label']), train, (test, test['label'])
        else:
            print("[log-load-snorkel-data] train file does not exist.")
            raise

    def read_all(self):
        data = self.read_semeval_2007_4()
        data = data.append(self.read_semeval_2010_8())
        data = data.append(self.read_event_causality())
        data = data.append(self.read_causal_timebank())
        data = data.append(self.read_story_lines())
        data = data.append(self.read_CaTeRS())
        data = data.append(self.read_because())
        # saving data
        data.to_csv(self.gold_causal_path)
        return data

    def _check_text_format(self, e1_span, e2_span, text):
        """
        check if a causal entry has a standard format
        :param e1_span:
        :param e2_span:
        :param text:
        :return:
        """
        return True if e1_span.strip() != "" and e2_span.strip() != "" and self.arg1_tag[0] in text and self.arg2_tag[0] in text else False

    def read_semeval_2007_4(self):
        """
        reading SemEval 2007 task 4 data - Relation 1 is cause-effect
        :return: pandas data frame of samples
        """

        data = []

        def extract_samples(all_docs, split):
            # each sample has three lines of information
            for idx, val in enumerate(all_docs):
                tmp = val.split(" ", 1)
                if tmp[0].isalnum():
                    sample_id = copy.deepcopy(tmp[0])
                    try:
                        sentence_text = copy.deepcopy(tmp[1])
                        text_string = tmp[1].replace("\"", "")

                        # extracting elements (nominals) of causal relation
                        e1_text = self._get_between_text("<e1>", "</e1>", sentence_text)
                        e2_text = self._get_between_text("<e2>", "</e2>", sentence_text)

                        tmp = all_docs[idx + 1].split(",")
                        if not ("true" in tmp[3] or "false" in tmp[3]):
                            tmp_label = tmp[2].replace(" ", "").replace("\"", "").split("=")
                        else:
                            tmp_label = tmp[3].replace(" ", "").replace("\"", "").split("=")

                        # label = 1, it's a causal samples, otherwise, it's not.
                        if tmp_label[1] == "true":
                            label = 1
                        else:
                            label = 0

                        # label_direction = 0 -> (e1, e2), otherwise, (e2, e1), is 1
                        if "e2" in tmp_label[0]:
                            label_direction = 0
                        else:
                            label_direction = 1

                        # replacing tags with standard tags
                        text_string = text_string.replace("<e1>", self.arg1_tag[0])
                        text_string = text_string.replace("</e1>", self.arg1_tag[1])
                        text_string = text_string.replace("<e2>", self.arg2_tag[0])
                        text_string = text_string.replace("</e2>", self.arg2_tag[1])

                        if self._check_text_format(e1_text, e2_text, text_string):
                            data.append([int(sample_id), e1_text, e2_text, text_string.replace('\n', ' '), label_direction, label,
                                         self.semeval_2007_4_code, "", split])
                    except Exception as e:
                        print("[datareader-semeval-log] Incorrect formatting! Details: " + str(e))

        # reading files
        with open(self.dir_path + 'SemEval2007_task4/task-4-training/relation-1-train.txt', mode='r',
                  encoding='cp1252') as train:
            train_content = train.readlines()

        # this is the test set
        with open(self.dir_path + 'SemEval2007_task4/task-4-scoring/relation-1-score.txt', mode='r',
                  encoding='cp1252') as key:
            key_content = key.readlines()

        extract_samples(train_content, 0)
        extract_samples(key_content, 2)

        # print("Total samples = " + str(len(data)))
        return pd.DataFrame(data, columns=self.scheme_columns)

    def read_semeval_2010_8(self):
        """
        reading SemEval 2010 task 8 data - Relation 1 is cause-effect
        :return: pandas data frame of samples
        """
        data = []

        def extract_samples(all_docs, split):
            # each sample has three lines of information
            for idx, val in enumerate(all_docs):
                tmp = val.split("\t")
                if tmp[0].isalnum():
                    try:
                        sentence_text = copy.deepcopy(tmp[1])
                        text_string = tmp[1].replace("\"", "")

                        # extracting elements (nominals) of causal relation
                        e1_text = self._get_between_text("<e1>", "</e1>", sentence_text)
                        e2_text = self._get_between_text("<e2>", "</e2>", sentence_text)

                        if "Cause-Effect" in all_docs[idx + 1]:
                            if "e1,e2" in all_docs[idx + 1]:
                                label_direction = 0
                            else:
                                label_direction = 1

                            # replacing tags with standard tags
                            text_string = text_string.replace("<e1>", self.arg1_tag[0])
                            text_string = text_string.replace("</e1>", self.arg1_tag[1])
                            text_string = text_string.replace("<e2>", self.arg2_tag[0])
                            text_string = text_string.replace("</e2>", self.arg2_tag[1])

                            if self._check_text_format(e1_text, e2_text, text_string):
                                data.append([tmp[0], e1_text, e2_text, text_string.replace('\n', ' '), label_direction, 1,
                                             self.semeval_2010_8_code, "", split])
                    except Exception as e:
                        print("[datareader-semeval-log] Incorrect formatting! Details: " + str(e))

        # reading files
        with open(self.dir_path + 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.txt', mode='r',
                  encoding='cp1252') as train:
            train_content = train.readlines()

        with open(self.dir_path + 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt',
                  mode='r', encoding='cp1252') as key:
            test_content = key.readlines()

        extract_samples(train_content, 0)
        extract_samples(test_content, 2)

        # print("Total samples = " + str(len(data)))
        return pd.DataFrame(data, columns=self.scheme_columns)

    def read_event_causality(self):

        def get_sentence_text(sentence_string):
            sen = ""
            for t in sentence_string.split(' '):
                sen += t.split('/')[0] + " "
            return sen.strip() + " "

        keys = {}

        def read_keys(files, split):
            """
            reading all dev and eval keys
            :return: a dictionary of list of dictionaries where key is the document id and list contains
            dictionaries with two keys, p1 and p2, respectively. p1 and p2 are predicates which are annotated arguments
            in causal relations.
            """
            for file in files:
                with open(file, 'r') as file:
                    data = file.read()
                    data = "<keys>" + data + "</keys>"

                tree = ET.fromstring(data)
                # there're two types of tags -> C: causal, R: related
                for child in tree:
                    doc_id = child.attrib['id']
                    doc_tags = []
                    tags = child.text.split("\n")
                    for tag in tags:
                        # reading only causal (C) tags
                        if tag.startswith("C"):
                            if "\t" in tag:
                                tag_var = tag.split("\t")
                            else:
                                tag_var = tag.split(" ")
                            doc_tags.append({'p1': tag_var[1], 'p2': tag_var[2], 'split': split})
                    keys[doc_id] = doc_tags

        folders = ['dev', 'eval']
        excluded_files = ['.DS_Store']
        data_path = self.dir_path + "EventCausalityData/"

        dev_keys_file = data_path + "keys/dev.keys"
        eval_keys_file = data_path + "keys/eval.keys"

        # reading all keys
        read_keys([dev_keys_file], 1)
        read_keys([eval_keys_file], 2)

        docs = {}
        for folder in folders:
            all_files = os.listdir(data_path + folder)
            for file in all_files:
                if file not in excluded_files:
                    try:
                        parser = ET.XMLParser(encoding="utf-8")
                        tree = ET.parse(data_path + folder + "/" + file, parser=parser)
                        root = tree.getroot()
                        doc_id = root.attrib['id']
                        sentences = {}
                        # TODO: use findall() instead of nested for-if loops
                        for child in root:
                            if child.tag == "P":
                                for cchild in child:
                                    if cchild.tag == "S3":
                                        text_id = cchild.attrib['id']
                                        sentences[text_id] = cchild.text
                        docs[doc_id] = sentences
                    except Exception as e:
                        print("[datareader-event-causality-log] error in parsing file. Details: " + str(e))

        data = []
        data_idx = 1
        # now that we have all the information in dictionaries, we create samples
        for key, values in keys.items():
            if key in docs:
                # each key is a doc id
                for value in values:
                    p1 = value['p1'].split('_')
                    p2 = value['p2'].split('_')
                    split = value['split']
                    if p1[0] == p2[0]:
                        sentence = ""
                        p1_token_idx = p1[1]
                        p2_token_idx = p2[1]
                        doc_sentences = docs[key]
                        tokens = doc_sentences[p1[0]].split(' ')
                        for i in range(len(tokens)):
                            token = tokens[i].split('/')[0]
                            # TODO: change it in a way that there's always e1 and then e2 in the documents and
                            #  direction label will make it clear which one is cause and which one is effect
                            if i == int(p1_token_idx):
                                sentence += self.arg1_tag[0] + token + self.arg1_tag[1] + " "
                                p1_token = copy.deepcopy(token)
                            elif i == int(p2_token_idx):
                                sentence += self.arg2_tag[0] + token + self.arg2_tag[1] + " "
                                p2_token = copy.deepcopy(token)
                            else:
                                sentence += token + " "

                        # TODO: check the label direction
                        if self._check_text_format(p1_token, p2_token, sentence):
                            data.append([data_idx, p1_token, p2_token, sentence.replace('\n', ' '), 1, 1, self.event_causality_code, "", split])
                            data_idx += 1
                    else:
                        s_idx = {}
                        label_direction = 0
                        if int(p2[0]) < int(p1[0]):
                            s_idx[1] = {'sen_index': int(p2[0]), 'token_index': int(p2[1]), 'token_label': self.arg2_tag}
                            s_idx[2] = {'sen_index': int(p1[0]), 'token_index': int(p1[1]), 'token_label': self.arg1_tag}
                            label_direction = 1
                        else:
                            s_idx[1] = {'sen_index': int(p1[0]), 'token_index': int(p1[1]), 'token_label': self.arg1_tag}
                            s_idx[2] = {'sen_index': int(p2[0]), 'token_index': int(p2[1]), 'token_label': self.arg2_tag}

                        sentence = ""

                        doc_sentences = docs[key]
                        for k, v in doc_sentences.items():
                            tokens = v.split(' ')
                            if int(k) == s_idx[1]['sen_index']:
                                for i in range(len(tokens)):
                                    token = tokens[i].split('/')[0]
                                    if i == int(s_idx[1]['token_index']):
                                        sentence += s_idx[1]['token_label'][0] + token + s_idx[1]['token_label'][1] + " "
                                        p1_token = copy.deepcopy(token)
                                    else:
                                        sentence += token + " "
                            elif int(k) == s_idx[2]['sen_index']:
                                for i in range(len(tokens)):
                                    token = tokens[i].split('/')[0]
                                    if i == int(s_idx[2]['token_index']):
                                        sentence += s_idx[2]['token_label'][0] + token + s_idx[2]['token_label'][1] + " "
                                        p2_token = copy.deepcopy(token)
                                    else:
                                        sentence += token + " "
                                break
                            elif s_idx[1]['sen_index'] < int(k) < s_idx[2]['sen_index']:
                                sentence += get_sentence_text(v)

                        if self._check_text_format(p1_token, p2_token, sentence):
                            data.append([data_idx, p1_token, p2_token, sentence.replace('\n', ' '), label_direction, 1, self.event_causality_code, "", split])
                            data_idx += 1

        # print("Total samples = " + str(len(data)))
        return pd.DataFrame(data, columns=self.scheme_columns)

    def read_causal_timebank(self):
        data_path = self.dir_path + "Causal-TimeBank/Causal-TimeBank-CAT"
        all_files = os.listdir(data_path)
        # parser = ET.XMLParser(encoding="utf-8")
        data_idx = 1
        data = []
        for file in all_files:
            tokens = []
            try:
                tree = ET.parse(data_path + "/" + file)
                root = tree.getroot()
            except Exception as e:
                print("file: " + str(file))
                print(str(e))
                print("---------")

            # [0] getting information of events
            events = {}
            for event in root.findall('Markables/EVENT'):
                events_ids = []
                for anchor in event:
                    events_ids.append(anchor.attrib['id'])
                events[event.attrib['id']] = events_ids

            for event in root.findall('Markables/TIMEX3'):
                events_ids = []
                for anchor in event:
                    events_ids.append(anchor.attrib['id'])
                events[event.attrib['id']] = events_ids

            for event in root.findall('Markables/C-SIGNAL'):
                events_ids = []
                for anchor in event:
                    events_ids.append(anchor.attrib['id'])
                events[event.attrib['id']] = events_ids

            # [1] getting list of tokens in sentence/doc
            for token in root.findall('token'):
                tokens.append([token.attrib['id'], token.text, token.attrib['number'], token.attrib['sentence']])

            # [2] getting list of causal links
            for link in root.findall('Relations/CLINK'):
                source_id = link[0].attrib['id']
                target_id = link[1].attrib['id']

                if 'c-signalID' in link.attrib:
                    signal_id = link.attrib['c-signalID']
                    signal_anchors = events[signal_id]

                # finding label direction
                if source_id > target_id:
                    label_direction = 1
                    s_id = target_id
                    t_id = source_id
                else:
                    label_direction = 0
                    s_id = source_id
                    t_id = target_id

                sentence = ""
                start_anchors = events[s_id]
                end_anchors = events[t_id]
                e1_tokens = ""
                e2_tokens = ""

                # finding start and end sentences indexes
                for i in range(len(tokens)):
                    if tokens[i][0] == start_anchors[0]:
                        s_sen_id = tokens[i][3]
                        break
                for i in range(len(tokens)):
                    if tokens[i][0] == end_anchors[0]:
                        t_sen_id = tokens[i][3]
                        break

                for i in range(len(tokens)):
                    token_id = tokens[i][0]
                    token_text = tokens[i][1]
                    token_sen_id = tokens[i][3]
                    if s_sen_id <= token_sen_id <= t_sen_id:
                        if token_id == start_anchors[0]:
                            sentence = sentence + self.arg1_tag[0]
                            for l in range(len(start_anchors)):
                                sentence += tokens[i+l][1] + " "
                                e1_tokens += tokens[i+l][1] + " "
                            sentence = sentence.strip() + self.arg1_tag[1] + " "

                        elif token_id == end_anchors[0]:
                            sentence = sentence + self.arg2_tag[0]
                            for l in range(len(end_anchors)):
                                sentence += tokens[i+l][1] + " "
                                e2_tokens += tokens[i+l][1] + " "
                            sentence = sentence.strip() + self.arg2_tag[1] + " "

                        elif token_id == signal_anchors[0]:
                            sentence += self.signal_tag[0]
                            for l in range(len(signal_anchors)):
                                sentence += tokens[i+l][1] + " "
                            sentence = sentence.strip() + self.signal_tag[1] + " "
                        else:
                            sentence += token_text + " "

                if self._check_text_format(e1_tokens, e2_tokens, sentence):
                    data.append([data_idx, e1_tokens, e2_tokens, sentence.replace('\n', ' '), label_direction, 1, self.causal_timebank_code, file, ""])
                    data_idx += 1

        # print("Total samples = " + str(len(data)))
        return pd.DataFrame(data, columns=self.scheme_columns)

    def read_story_lines(self):

        def get_e_info(e_list):
            """
            reading the event-event pair info
            :param e_list: Event-Event Pair column value in ground truth csv file
            :return:
            """
            e = e_list.split('_')
            if len(e) == 3:
                e_text = e[0]
                e_start = e[1]
                e_end = e[2]
            else:
                e_start = e[len(e) - 2]
                e_end = e[len(e) - 1]
                e_text = ""
                for i in range(len(e) - 2):
                    e_text += e[i] + " "
                e_text = e_text.strip()
            return e_text, int(e_start), int(e_end)

        expert_labels_file = self.dir_path + "Crowdsourcing-StoryLines/old/data/ground_truth/main_experiments_ground_truth.csv"
        documents_path = self.dir_path + "Crowdsourcing-StoryLines/old/EventStoryLine/annotated_data/v1.0"
        # documents_path = self.dir_path + "Crowdsourcing-StoryLines/EventStoryLine/ECB+_LREC2014/ECB+"
        ground_truth = pd.read_csv(expert_labels_file)
        all_files = os.listdir(documents_path)

        # creating a dictionary of all documents
        docs_info = {}
        for file in all_files:
            if ".txt" not in file and ".DS_Store" not in file:
                docs = os.listdir(documents_path + "/" + file)
                for doc in docs:
                    if ".xml" in doc:
                        # parse the doc to retrieve info of sentences
                        tree = ET.parse(documents_path + "/" + file + "/" + doc)
                        root = tree.getroot()
                        current_sentence = ""
                        current_sentence_id = -1
                        docs_info[doc.strip(".xml")] = {}
                        for token in root.findall('token'):
                            if token.attrib['sentence'] != current_sentence_id and current_sentence_id != -1:
                                docs_info[doc.strip(".xml")][current_sentence_id] = current_sentence.strip()
                                current_sentence = ""
                            current_sentence_id = token.attrib['sentence']
                            current_sentence += token.text + " "
                        # saving the last sentence
                        docs_info[doc.strip(".xml")][current_sentence_id] = current_sentence.strip()
        # print("Total documents: " + str(len(docs_info)))

        data_idx = 1
        data = []
        err = 0
        for index, row in ground_truth.iterrows():
            tmp = row["Document Id"].split('_')
            if len(tmp) == 3:
                doc_id = tmp[0] + '_' + tmp[1].strip(".xml")
                sentence_id = tmp[2]
                if "-r-" in row["Event-Event Pair"]:
                    label_direction = 1
                    ee = row["Event-Event Pair"].split("-r-")
                else:
                    label_direction = 0
                    ee = row["Event-Event Pair"].split("--")

                try:
                    e1_tokens, e1_start, e1_end = get_e_info(ee[0])
                    e2_tokens, e2_start, e2_end = get_e_info(ee[1])

                    sentence = docs_info[doc_id][sentence_id]
                    sentence = self._add_events_tags(sentence, e1_start, e1_end, e2_start, e2_end)

                    if self._check_text_format(e1_tokens, e2_tokens, sentence):
                        data.append([data_idx, e1_tokens, e2_tokens, sentence.replace('\n', ' '), label_direction, 1, self.storyline_code, "", ""])
                        data_idx += 1
                except Exception as e:
                    print(str(e), doc_id, sentence_id)
                    err += 1
        # print("Total samples = " + str(len(data)))
        if err > 0:
            print("[datareader-story-line-log] err: " + str(err))
        return pd.DataFrame(data, columns=self.scheme_columns)

    def read_CaTeRS(self):

        folders_path = self.dir_path + "caters/caters_evaluation/"
        data = []

        def _process_args(doc_info, doc_txt, arg_1, arg_2):

            e1_args = (' '.join(doc_info[arg_1].split(' ')[1:]).strip()).split(';')
            e2_args = (' '.join(doc_info[arg_2].split(' ')[1:]).strip()).split(';')

            df_columns = ['start', 'end', 'start_tag', 'end_tag']
            df = pd.DataFrame(columns=df_columns)

            for item in e1_args:
                arg = item.split(' ')
                df = df.append(pd.DataFrame(
                    [[int(arg[0]), int(arg[1]), self.arg1_tag[0], self.arg1_tag[1]]],
                    columns=df_columns
                ), ignore_index=True)

            for item in e2_args:
                arg = item.split(' ')
                df = df.append(pd.DataFrame(
                    [[int(arg[0]), int(arg[1]), self.arg2_tag[0], self.arg2_tag[1]]],
                    columns=df_columns
                ), ignore_index=True)

            df = df.sort_values(by=['start'])
            tagged_sentence = self._add_sentence_tags(doc_txt, df)

            # cutting extra sentences from the tagged_sentence
            sentences = tagged_sentence.split("***")
            trimmed_doc = ""
            for sent in sentences:
                if self.arg1_tag[0] in sent or self.arg2_tag[0] in sent:
                    trimmed_doc += sent.strip() + " "

            e1_txt = ""
            e2_txt = ""
            for index, row in df.iterrows():
                if self.arg1_tag[0] == row['start_tag']:
                    e1_txt += doc_txt[row['start']:row['end']] + " "
                elif self.arg2_tag[0] == row['start_tag']:
                    e2_txt += doc_txt[row['start']:row['end']] + " "

            return e1_txt.strip(), e2_txt.strip(), trimmed_doc.strip()

        def extract_samples(folders, split, data_idx, err_idx):
            for folder in folders:
                docs_path = folders_path + "/" + folder
                docs = os.listdir(docs_path)
                for doc in docs:
                    doc_info = {}
                    if ".ann" in doc:
                        with open(docs_path + "/" + doc, 'r') as f:
                            lines = f.readlines()

                            # reading all tags information
                            for line in lines:
                                line_cols = line.split('\t')
                                doc_info[line_cols[0]] = line_cols[1]

                            # reading the corresponding text file for the .ann file
                            with open(docs_path + "/" + doc.strip(".ann") + ".txt", 'r') as f:
                                doc_string = f.read()

                            # iterate through causal tags
                            for key, value in doc_info.items():
                                try:
                                    if "CAUSE" in value:
                                        args = value.split(' ')
                                        arg1 = args[1].split(':')[1]
                                        arg2 = args[2].split(':')[1]

                                        e1_tokens, e2_tokens, sentence = _process_args(doc_info, doc_string, arg1, arg2)

                                        # adding new record
                                        if self._check_text_format(e1_tokens, e2_tokens, sentence):
                                            data.append([data_idx, e1_tokens, e2_tokens, sentence.replace('\n', ' '), 0, 1, self.caters_code, doc, split])
                                            data_idx += 1
                                except Exception as e:
                                    print("[datareader-caters-log] Error in processing causal relation info. Details: " + str(e))
                                    err_idx += 1
            return data_idx, err_idx

        data_idx = 1
        err_idx = 0
        data_idx, err_idx = extract_samples(["dev"], 1, data_idx, err_idx)
        data_idx, err_idx = extract_samples(["train"], 0, data_idx, err_idx)

        # print("Total samples = " + str(len(data)))
        if err_idx > 0:
            print("[datareader-CaTeRS-log] err: " + str(err_idx))
        return pd.DataFrame(data, columns=self.scheme_columns)

    def read_because(self):
        """
        reading BECAUSE v2.1 Data
        :return:
        """

        nlp = spacy.load("en_core_web_sm")

        def _get_doc_sentences(text):
            """
            splitting a document into its sentences
            :param text: input documents (can be of any length from one sentence to multiple paragraphs)
            :return:
            """
            sentences = []
            doc = nlp(text)
            sents = list(doc.sents)
            for sen in sents:
                sentences.append(sen.text)
            return sentences

        def _remove_extra_sentences(doc_text):
            sentences = _get_doc_sentences(doc_text)
            tag_set = [self.arg1_tag[0].replace('<', '').replace('</', '').replace('>', ''),
                       self.arg1_tag[1].replace('<', '').replace('</', '').replace('>', ''),
                       self.arg2_tag[0].replace('<', '').replace('</', '').replace('>', ''),
                       self.arg2_tag[1].replace('<', '').replace('</', '').replace('>', ''),
                       self.signal_tag[0].replace('<', '').replace('</', '').replace('>', ''),
                       self.signal_tag[1].replace('<', '').replace('</', '').replace('>', '')]

            s_index = -1
            e_index = -1

            for i in range(len(sentences)):
                if any(tag in sentences[i] for tag in tag_set):
                    s_index = i
                    break

            for i in range(len(sentences)-1, -1, -1):
                if any(tag in sentences[i] for tag in tag_set):
                    e_index = i
                    break

            try:
                assert s_index != -1 and e_index != -1
                assert s_index <= e_index
            except AssertionError as error:
                print("[datareader-because-log] Details: " + str(error))

            before_after_window = 1
            if s_index - before_after_window < 0:
                s_index = s_index - (abs(s_index - before_after_window))
            else:
                s_index = s_index - before_after_window

            if (e_index + before_after_window) > len(sentences):
                e_index = e_index + (abs(len(sentences) - e_index))
            else:
                e_index = e_index + before_after_window

            doc_cleaned = (" ".join(sentences[s_index:e_index+1])).replace(' >', '>')

            return doc_cleaned.replace('< ', '<').replace('</ ', '</').replace('< /', '</').replace(' >', '>')

        def _process_args(doc, cause_tag, effect_tag, signal_tag):
            e1 = ' '.join(doc_info[cause_tag].split(' ')[1:]).strip()
            e2 = ' '.join(doc_info[effect_tag].split(' ')[1:]).strip()
            sig = ' '.join(doc_info[signal_tag].split(' ')[1:]).strip()

            e1_args = e1.split(';')
            e2_args = e2.split(';')
            sig_args = sig.split(';')

            df_columns = ['start', 'end', 'start_tag', 'end_tag']
            df = pd.DataFrame(columns=df_columns)

            for arg in e1_args:
                df = df.append(pd.DataFrame(
                    [[int(arg.split(' ')[0]), int(arg.split(' ')[1]), self.arg1_tag[0], self.arg1_tag[1]]],
                    columns=df_columns
                ), ignore_index=True)

            for arg in e2_args:
                df = df.append(pd.DataFrame(
                    [[int(arg.split(' ')[0]), int(arg.split(' ')[1]), self.arg2_tag[0], self.arg2_tag[1]]],
                    columns=df_columns
                ), ignore_index=True)

            for arg in sig_args:
                df = df.append(pd.DataFrame(
                    [[int(arg.split(' ')[0]), int(arg.split(' ')[1]), self.signal_tag[0], self.signal_tag[1]]],
                    columns=df_columns
                ), ignore_index=True)

            df = df.sort_values('start')
            tagged_sentence = self._add_sentence_tags(doc, df)

            e1_text = ""
            e2_text = ""
            for index, row in df.iterrows():
                if self.arg1_tag[0] == row['start_tag']:
                    e1_text += doc[row['start']:row['end']] + " "
                elif self.arg2_tag[0] == row['start_tag']:
                    e2_text += doc[row['start']:row['end']] + " "

            return e1_text.strip(), e2_text.strip(), tagged_sentence.strip()

        # for NYT and PTB LDC subscription is needed to get access to the raw text.
        folders = ["CongressionalHearings", "MASC"]
        folders_path = self.dir_path + "BECAUSE-2.1/"

        data_idx = 1
        data = []
        err = 0

        for folder in folders:
            docs_path = folders_path + folder
            docs = os.listdir(docs_path)
            for doc in docs:
                doc_info = {}
                if ".ann" in doc:
                    with open(docs_path + "/" + doc, 'r') as f:
                        lines = f.readlines()

                        # reading all tags information from .ann file
                        for line in lines:
                            line_cols = line.split('\t')
                            doc_info[line_cols[0]] = line_cols[1]

                    # reading the corresponding text file for the .ann file
                    with open(docs_path + "/" + doc.strip(".ann") + ".txt", 'r') as f:
                        doc_string = f.read()

                    # now, reading causal relations info
                    for key, value in doc_info.items():
                        try:
                            # check if the relation has all the arguments
                            value = value.replace('\n', '')

                            # causal samples
                            if ("Consequence" in value or "Purpose" in value or "Motivation" in value) and "Effect" in value and "Cause" in value:
                                args = value.split(' ')
                                signal_tag = args[0].split(':')[1]
                                if "Cause" in args[1]:
                                    cause_tag = args[1].split(':')[1]
                                    effect_tag = args[2].split(':')[1]
                                else:
                                    cause_tag = args[2].split(':')[1]
                                    effect_tag = args[1].split(':')[1]

                                e1_tokens, e2_tokens, doc_text = _process_args(doc_string, cause_tag, effect_tag, signal_tag)

                                if self._check_text_format(e1_tokens, e2_tokens, _remove_extra_sentences(doc_text)):
                                    data.append([data_idx, e1_tokens, e2_tokens, _remove_extra_sentences(doc_text).replace('\n', ' '), 0, 1, self.because_code, doc, ""])
                                    data_idx += 1

                            # non-causal samples
                            elif ("NonCausal" in value) and "Arg0" in value and "Arg1" in value:
                                args = value.split(' ')
                                signal_tag = args[0].split(':')[1]
                                arg0_tag = args[1].split(':')[1]
                                arg1_tag = args[2].split(':')[1]
                                e1_tokens, e2_tokens, doc_text = _process_args(doc_string, arg0_tag, arg1_tag, signal_tag)

                                if self._check_text_format(e1_tokens, e2_tokens, _remove_extra_sentences(doc_text)):
                                    data.append([data_idx, e1_tokens, e2_tokens, _remove_extra_sentences(doc_text).replace('\n', ' '), 0, 0, self.because_code, doc, ""])
                                    data_idx += 1

                        except Exception as e:
                            print("[datareader-because-log] Error in processing causal relation info. Details: " + str(e))
                            err += 1
        # print("Total samples = " + str(len(data)))
        if err > 0:
            print("[datareader-because-log] # of errors: " + str(err))
        return pd.DataFrame(data, columns=self.scheme_columns)

    def _add_events_tags(self, doc_text, e1_start, e1_end, e2_start, e2_end):
        """
        adding events tags to the doc string
        :param doc_text:
        :param e1_start:
        :param e1_end:
        :param e2_start:
        :param e2_end:
        :return:
        """
        return doc_text[:e1_start] + self.arg1_tag[0] + doc_text[e1_start:e1_end] + self.arg1_tag[1] + \
               doc_text[e1_end:e2_start] + self.arg2_tag[0] + doc_text[e2_start:e2_end] + self.arg2_tag[1] + \
               doc_text[e2_end:]

    def read_causal_connectives(self):
        """
        reading causal connectives other than those from well-known resources
        :return:
        """
        contingency_cause_word = {}
        contingency_cause_count = {}
        contingency_cause_type = {}  # 0: explicit, 1: implicit, 2: explicit and implicit
        contingency_cause_temporal = {}  # 0: not temporal, 1: temporal and causal
        contingency_cause_flag = {}  # 0: reason, 1: result, 2: reason and result

        contingency_cause_word[0] = "cause"
        contingency_cause_count[0] = -1
        contingency_cause_type[0] = 0
        contingency_cause_temporal[0] = 0
        contingency_cause_flag[0] = 0

        df = pd.DataFrame(columns=self.connective_columns)

        for key, value in contingency_cause_word.items():
            df = df.append(pd.DataFrame([[value,
                                          contingency_cause_count[key],
                                          contingency_cause_type[key],
                                          contingency_cause_temporal[key],
                                          contingency_cause_flag[key]]], columns=self.connective_columns), ignore_index=True)

        return df

    def read_PDTB_causal_connectors(self, n_min=0):
        """
        getting a list of all contingency cause/reason, explicit/implicit connectors from Penn Discourse TreeBank (PDTB) 3.0
        :return:
        """
        # TODO: putting information in dictionaries not by hard coding the indexes.
        # list of causal connectors based on PDTB 3.0
        # commented numbers above the lines are the frequency of words when they are "implicit" connectors
        contingency_cause_word = {}
        contingency_cause_count = {}
        contingency_cause_type = {}  # 0: explicit, 1: implicit, 2: explicit and implicit
        contingency_cause_temporal = {}  # 0: not temporal, 1: temporal and causal
        contingency_cause_flag = {}  # 0: reason, 1: result, 2: reason and result

        # --------------------------------
        # explicit and implicit connectors
        # --------------------------------
        contingency_cause_word[0] = "about"
        contingency_cause_count[0] = 2
        contingency_cause_type[0] = 0
        contingency_cause_temporal[0] = 0
        contingency_cause_flag[0] = 0

        # 86
        contingency_cause_word[1] = "accordingly"
        contingency_cause_count[1] = 91
        contingency_cause_type[1] = 2
        contingency_cause_temporal[1] = 0
        contingency_cause_flag[1] = 1

        contingency_cause_word[2] = "after"
        contingency_cause_count[2] = 56
        contingency_cause_type[2] = 0
        contingency_cause_temporal[2] = 1
        contingency_cause_flag[2] = 0

        # 337
        contingency_cause_word[3] = "as"
        contingency_cause_count[3] = 686
        contingency_cause_type[3] = 2
        contingency_cause_temporal[3] = 1
        contingency_cause_flag[3] = 0  # because it's "result" in only 1 case

        # 761
        contingency_cause_word[4] = "as a result"
        contingency_cause_count[4] = 839
        contingency_cause_type[4] = 2
        contingency_cause_temporal[4] = 0
        contingency_cause_flag[4] = 1

        # 2059
        contingency_cause_word[5] = "because"
        contingency_cause_count[5] = 2892
        contingency_cause_type[5] = 2
        contingency_cause_temporal[5] = 0
        contingency_cause_flag[5] = 0

        contingency_cause_word[6] = "by"
        contingency_cause_count[6] = 10
        contingency_cause_type[6] = 0
        contingency_cause_temporal[6] = 0
        contingency_cause_flag[6] = 0

        contingency_cause_word[7] = "by then"
        contingency_cause_count[7] = 1
        contingency_cause_type[7] = 0
        contingency_cause_temporal[7] = 1
        contingency_cause_flag[7] = 0

        # 213
        contingency_cause_word[8] = "consequently"
        contingency_cause_count[8] = 223
        contingency_cause_type[8] = 2
        contingency_cause_temporal[8] = 0
        contingency_cause_flag[8] = 1

        contingency_cause_word[9] = "due to"
        contingency_cause_count[9] = 1
        contingency_cause_type[9] = 0
        contingency_cause_temporal[9] = 0
        contingency_cause_flag[9] = 0

        # 1
        contingency_cause_word[10] = "finally"
        contingency_cause_count[10] = 2
        contingency_cause_type[10] = 2
        contingency_cause_temporal[10] = 1
        contingency_cause_flag[10] = 1

        contingency_cause_word[11] = "for"
        contingency_cause_count[11] = 35
        contingency_cause_type[11] = 2
        contingency_cause_temporal[11] = 0
        contingency_cause_flag[11] = 0

        contingency_cause_word[12] = "from"
        contingency_cause_count[12] = 2
        contingency_cause_type[12] = 0
        contingency_cause_temporal[12] = 0
        contingency_cause_flag[12] = 0

        # 16
        contingency_cause_word[13] = "hence"
        contingency_cause_count[13] = 21
        contingency_cause_type[13] = 2
        contingency_cause_temporal[13] = 0
        contingency_cause_flag[13] = 1

        # 4
        contingency_cause_word[14] = "in"
        contingency_cause_count[14] = 5
        contingency_cause_type[14] = 2
        contingency_cause_temporal[14] = 0
        contingency_cause_flag[14] = 0

        # 4
        contingency_cause_word[15] = "in the end"
        contingency_cause_count[15] = 6
        contingency_cause_type[15] = 2
        contingency_cause_temporal[15] = 0
        contingency_cause_flag[15] = 1

        # 4
        contingency_cause_word[16] = "indeed"
        contingency_cause_count[16] = 5
        contingency_cause_type[16] = 2
        contingency_cause_temporal[16] = 0
        contingency_cause_flag[16] = 2

        # 2
        contingency_cause_word[17] = "insofar as"
        contingency_cause_count[17] = 3
        contingency_cause_type[17] = 2
        contingency_cause_temporal[17] = 0
        contingency_cause_flag[17] = 0

        contingency_cause_word[18] = "not only because of"
        contingency_cause_count[18] = 1
        contingency_cause_type[18] = 0
        contingency_cause_temporal[18] = 0
        contingency_cause_flag[18] = 0

        contingency_cause_word[19] = "now that"
        contingency_cause_count[19] = 19
        contingency_cause_type[19] = 0
        contingency_cause_temporal[19] = 1
        contingency_cause_flag[19] = 0

        contingency_cause_word[20] = "on"
        contingency_cause_count[20] = 1
        contingency_cause_type[20] = 0
        contingency_cause_temporal[20] = 0
        contingency_cause_flag[20] = 0

        contingency_cause_word[21] = "once"
        contingency_cause_count[21] = 7
        contingency_cause_type[21] = 0
        contingency_cause_temporal[21] = 1
        contingency_cause_flag[21] = 0

        # 206
        contingency_cause_word[22] = "since"
        contingency_cause_count[22] = 313
        contingency_cause_type[22] = 2
        contingency_cause_temporal[22] = 1
        contingency_cause_flag[22] = 0

        # 989
        contingency_cause_word[23] = "so"
        contingency_cause_count[23] = 1211
        contingency_cause_type[23] = 2
        contingency_cause_temporal[23] = 0
        contingency_cause_flag[23] = 2

        # 2
        contingency_cause_word[24] = "so that"
        contingency_cause_count[24] = 12
        contingency_cause_type[24] = 2
        contingency_cause_temporal[24] = 0
        contingency_cause_flag[24] = 1

        # 3
        contingency_cause_word[25] = "then"
        contingency_cause_count[25] = 15
        contingency_cause_type[25] = 2
        contingency_cause_temporal[25] = 1
        contingency_cause_flag[25] = 1

        contingency_cause_word[26] = "thereby"
        contingency_cause_count[26] = 9
        contingency_cause_type[26] = 0
        contingency_cause_temporal[26] = 0
        contingency_cause_flag[26] = 1

        # 326
        contingency_cause_word[27] = "therefore"
        contingency_cause_count[27] = 352
        contingency_cause_type[27] = 2
        contingency_cause_temporal[27] = 0
        contingency_cause_flag[27] = 1

        # 388
        contingency_cause_word[28] = "thus"
        contingency_cause_count[28] = 499
        contingency_cause_type[28] = 2
        contingency_cause_temporal[28] = 0
        contingency_cause_flag[28] = 1

        # 3
        contingency_cause_word[29] = "ultimately"
        contingency_cause_count[29] = 5
        contingency_cause_type[29] = 2
        contingency_cause_temporal[29] = 1
        contingency_cause_flag[29] = 2

        contingency_cause_word[30] = "upon"
        contingency_cause_count[30] = 3
        contingency_cause_type[30] = 0
        contingency_cause_temporal[30] = 1
        contingency_cause_flag[30] = 0

        contingency_cause_word[31] = "when"
        contingency_cause_count[31] = 179
        contingency_cause_type[31] = 0
        contingency_cause_temporal[31] = 1
        contingency_cause_flag[31] = 0  # because it's "result" in only 3 cases.

        # 2
        contingency_cause_word[32] = "with"
        contingency_cause_count[32] = 111
        contingency_cause_type[32] = 2
        contingency_cause_temporal[32] = 0
        contingency_cause_flag[32] = 0

        contingency_cause_word[33] = "without"
        contingency_cause_count[33] = 2
        contingency_cause_type[33] = 0
        contingency_cause_temporal[33] = 0
        contingency_cause_flag[33] = 2

        # 2
        contingency_cause_word[34] = "and"
        contingency_cause_count[34] = 11
        contingency_cause_type[34] = 2
        contingency_cause_temporal[34] = 0
        contingency_cause_flag[34] = 2

        # 66
        contingency_cause_word[35] = "because of"
        contingency_cause_count[35] = 78
        contingency_cause_type[35] = 2
        contingency_cause_temporal[35] = 0
        contingency_cause_flag[35] = 0

        # 46
        contingency_cause_word[36] = "given"
        contingency_cause_count[36] = 52
        contingency_cause_type[36] = 2
        contingency_cause_temporal[36] = 0
        contingency_cause_flag[36] = 0

        # --------------------
        # implicit connectors
        # --------------------

        contingency_cause_word[37] = "but"
        contingency_cause_count[37] = 7
        contingency_cause_type[37] = 1
        contingency_cause_temporal[37] = 0
        contingency_cause_flag[37] = 1

        contingency_cause_word[38] = "although"
        contingency_cause_count[38] = 1
        contingency_cause_type[38] = 1
        contingency_cause_temporal[38] = 0
        contingency_cause_flag[38] = 0

        contingency_cause_word[39] = "as a result of"
        contingency_cause_count[39] = 100
        contingency_cause_type[39] = 1
        contingency_cause_temporal[39] = 0
        contingency_cause_flag[39] = 0

        contingency_cause_word[40] = "as a result of being"
        contingency_cause_count[40] = 84
        contingency_cause_type[40] = 1
        contingency_cause_temporal[40] = 0
        contingency_cause_flag[40] = 0

        contingency_cause_word[41] = "as a result of having"
        contingency_cause_count[41] = 1
        contingency_cause_type[41] = 1
        contingency_cause_temporal[41] = 0
        contingency_cause_flag[41] = 0

        contingency_cause_word[42] = "because it was"
        contingency_cause_count[42] = 1
        contingency_cause_type[42] = 1
        contingency_cause_temporal[42] = 0
        contingency_cause_flag[42] = 0

        contingency_cause_word[43] = "thus being"
        contingency_cause_count[43] = 1
        contingency_cause_type[43] = 1
        contingency_cause_temporal[43] = 0
        contingency_cause_flag[43] = 1

        contingency_cause_word[44] = "because of being"
        contingency_cause_count[44] = 4
        contingency_cause_type[44] = 1
        contingency_cause_temporal[44] = 0
        contingency_cause_flag[44] = 0

        contingency_cause_word[45] = "for example"
        contingency_cause_count[45] = 1
        contingency_cause_type[45] = 1
        contingency_cause_temporal[45] = 0
        contingency_cause_flag[45] = 0

        contingency_cause_word[46] = "for one thing"
        contingency_cause_count[46] = 1
        contingency_cause_type[46] = 1
        contingency_cause_temporal[46] = 0
        contingency_cause_flag[46] = 0

        contingency_cause_word[47] = "for the reason that"
        contingency_cause_count[47] = 1
        contingency_cause_type[47] = 1
        contingency_cause_temporal[47] = 0
        contingency_cause_flag[47] = 0

        contingency_cause_word[48] = "given that"
        contingency_cause_count[48] = 2
        contingency_cause_type[48] = 1
        contingency_cause_temporal[48] = 0
        contingency_cause_flag[48] = 0

        contingency_cause_word[49] = "however"
        contingency_cause_count[49] = 1
        contingency_cause_type[49] = 1
        contingency_cause_temporal[49] = 0
        contingency_cause_flag[49] = 0

        # 2
        contingency_cause_word[50] = "in fact"
        contingency_cause_count[50] = 5
        contingency_cause_type[50] = 1
        contingency_cause_temporal[50] = 0
        contingency_cause_flag[50] = 0

        # 1
        contingency_cause_word[51] = "in other words"
        contingency_cause_count[51] = 2
        contingency_cause_type[51] = 1
        contingency_cause_temporal[51] = 0
        contingency_cause_flag[51] = 2

        # 1
        contingency_cause_word[52] = "in short"
        contingency_cause_count[52] = 3
        contingency_cause_type[52] = 1
        contingency_cause_temporal[52] = 0
        contingency_cause_flag[52] = 2

        contingency_cause_word[53] = "inasmuch as"
        contingency_cause_count[53] = 12
        contingency_cause_type[53] = 1
        contingency_cause_temporal[53] = 0
        contingency_cause_flag[53] = 0

        contingency_cause_word[54] = "it is because"
        contingency_cause_count[54] = 3
        contingency_cause_type[54] = 1
        contingency_cause_temporal[54] = 0
        contingency_cause_flag[54] = 0

        contingency_cause_word[55] = "on account of being"
        contingency_cause_count[55] = 1
        contingency_cause_type[55] = 1
        contingency_cause_temporal[55] = 0
        contingency_cause_flag[55] = 0

        # 7
        contingency_cause_word[56] = "so as"
        contingency_cause_count[56] = 8
        contingency_cause_type[56] = 1
        contingency_cause_temporal[56] = 0
        contingency_cause_flag[56] = 2

        contingency_cause_word[57] = "specifically"
        contingency_cause_count[57] = 1
        contingency_cause_type[57] = 1
        contingency_cause_temporal[57] = 0
        contingency_cause_flag[57] = 0

        # 3
        contingency_cause_word[58] = "that is"
        contingency_cause_count[58] = 4
        contingency_cause_type[58] = 1
        contingency_cause_temporal[58] = 0
        contingency_cause_flag[58] = 2

        contingency_cause_word[59] = "this is because"
        contingency_cause_count[59] = 1
        contingency_cause_type[59] = 1
        contingency_cause_temporal[59] = 0
        contingency_cause_flag[59] = 0

        contingency_cause_word[60] = "as a consequence"
        contingency_cause_count[60] = 2
        contingency_cause_type[60] = 1
        contingency_cause_temporal[60] = 0
        contingency_cause_flag[60] = 1

        contingency_cause_word[61] = "as it turns out"
        contingency_cause_count[61] = 1
        contingency_cause_type[61] = 1
        contingency_cause_temporal[61] = 0
        contingency_cause_flag[61] = 1

        contingency_cause_word[62] = "to this end"
        contingency_cause_count[62] = 1
        contingency_cause_type[62] = 1
        contingency_cause_temporal[62] = 0
        contingency_cause_flag[62] = 1

        contingency_cause_word[63] = "as such"
        contingency_cause_count[63] = 5
        contingency_cause_type[63] = 1
        contingency_cause_temporal[63] = 0
        contingency_cause_flag[63] = 1

        contingency_cause_word[64] = "because of that"
        contingency_cause_count[64] = 4
        contingency_cause_type[64] = 1
        contingency_cause_temporal[64] = 0
        contingency_cause_flag[64] = 1

        contingency_cause_word[65] = "for that reason"
        contingency_cause_count[65] = 2
        contingency_cause_type[65] = 1
        contingency_cause_temporal[65] = 0
        contingency_cause_flag[65] = 1

        contingency_cause_word[66] = "furthermore"
        contingency_cause_count[66] = 1
        contingency_cause_type[66] = 1
        contingency_cause_temporal[66] = 0
        contingency_cause_flag[66] = 1

        contingency_cause_word[67] = "in response"
        contingency_cause_count[67] = 2
        contingency_cause_type[67] = 1
        contingency_cause_temporal[67] = 0
        contingency_cause_flag[67] = 1

        df = pd.DataFrame(columns=self.connective_columns)

        for key, value in contingency_cause_word.items():
            if contingency_cause_count[key] > n_min:
                df = df.append(pd.DataFrame([[value,
                                              contingency_cause_count[key],
                                              contingency_cause_type[key],
                                              contingency_cause_temporal[key],
                                              contingency_cause_flag[key]]], columns=self.connective_columns), ignore_index=True)

        return df

    def read_wordnet_cause_effect(self):
        """
        finding all the synsets involved in a causal relation in wordnet
        :return:
        """
        all_synsets = list(wn.all_synsets())
        cause_list = []
        effect_list = []
        for i in range(len(all_synsets)):
            if len(all_synsets[i].causes()) > 0:
                cause_list.extend([lemma.replace('_', ' ') for lemma in all_synsets[i].lemma_names()])
                for synset in all_synsets[i].causes():
                    effect_list.extend([lemma.replace('_', ' ') for lemma in synset.lemma_names()])
        return self._lemmatize_list(cause_list), self._lemmatize_list(effect_list)

    def read_causal_frames_lus(self):
        """
        reading lexical units (lus) which evoke causal frames in FrameNet
        :return: a dictionary of lexical units with a set of POS tags as values for each lexical unit
        """
        causal_frames = []
        cause_fe = ["cause", "reason"]
        effect_fe = ["effect", "result"]
        frames = fn.frames()
        for frame in frames:
            f_elements = [x.name.lower() for x in frame.FE.values()]

            if any(c in f_elements for c in cause_fe) and any(e in f_elements for e in effect_fe):
                causal_frames.append(frame)

        units = {}
        for frame in causal_frames:
            for unit in frame.lexUnit:
                u = unit.split('.')
                if len(u) == 2 and u[1] in ["v", "n"] and not any(c in u[0] for c in ["(", "["]):
                    if u[1] == "v":
                        unit_pos = "VERB"
                    else:
                        unit_pos = "NOUN"
                    # lemmatizing the units
                    unit_doc = self.nlp(u[0])
                    unit_lemmas = ' '.join([word.lemma_.lower() for word in unit_doc])
                    if unit_pos not in units:
                        units[unit_pos] = [unit_lemmas.strip()]
                    else:
                        units[unit_pos].append(unit_lemmas.strip())
        return units

    def read_verbnet_causal_verbs(self):
        """
        reading members of verb classes with causal thematic
        :return:
        """
        causal_verbs = []
        verb_classes = vn.classids()

        causal_semantic_roles = ["cause", "result", "stimulus"]

        for c in verb_classes:
            # check if the class has a CAUSAL theme
            verb_class = vn.vnclass(c)
            class_themes = vn.themroles(verb_class)
            for theme in class_themes:
                if any(sr in theme['type'].lower() for sr in causal_semantic_roles):
                    # adding class members to the list
                    class_members = vn.pprint_members(verb_class).replace('\n', '').split(' ')[1:]
                    causal_verbs.extend(class_members)
                    break

        return self._lemmatize_list(causal_verbs)

    def _lemmatize_list(self, words):
        """
        lemmatize words in a list
        :param words:
        :return:
        """
        words_lemmas = set()
        for word in words:
            # lemmatizing the words
            word_doc = self.nlp(word.replace('_', ' '))
            word_lemmas = ' '.join([w.lemma_.lower() for w in word_doc])
            if word_lemmas != "":
                words_lemmas.add(word_lemmas.strip())
        return words_lemmas

    @staticmethod
    def _get_between_text(str_1, str_2, orig_text):
        result = re.search(str_1 + "(.*)" + str_2, orig_text)
        return result.group(1)

    @staticmethod
    def _add_sentence_tags(doc_text, tags):
        """
        getting a data frame of tags and a document string, this method adds tags to the document text.
        :param doc_text:
        :param tags:
        :return:
        """

        # reset data frame's indexes since they may have changed after sorting
        tags = tags.reset_index(drop=True)

        all_sen = [(doc_text[:tags.iloc[0]['start']]).strip()]
        for index, row in tags.iterrows():
            if index != len(tags) - 1:
                all_sen.append((row['start_tag'] + doc_text[row['start']:row['end']] + row['end_tag'] + doc_text[row['end']:tags.iloc[index+1]['start']]).strip())

        all_sen.append((tags.iloc[len(tags)-1]['start_tag'] + doc_text[tags.iloc[len(tags)-1]['start']:tags.iloc[len(tags)-1]['end']] + tags.iloc[len(tags)-1]['end_tag'] + doc_text[tags.iloc[len(tags)-1]['end']:]).strip())

        sentence = ' '.join(s for s in all_sen).strip()

        return sentence