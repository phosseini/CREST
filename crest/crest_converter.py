import re
import os
import sys
import copy
import spacy
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from nltk.corpus import framenet as fn
from nltk.corpus import verbnet as vn
from nltk.corpus import wordnet as wn


class CrestConverter:
    def __init__(self):
        """
        idx = {'span1': [], 'span2': [], 'signal': []} -> indexes of span1, span2, and signal tokens in context
        label = 0 -> non-causal, 1: [span1, span2] -> cause-effect, 2: [span1, span2] -> effect-cause
        split -> 0: train, 1: dev, test: 2
        """
        root_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
        sys.path.insert(0, root_path)
        self.dir_path = root_path + "/data/causal/"
        self.scheme_columns = ['id', 'span1', 'span2', 'context', 'idx', 'label', 'source', 'ann_file', 'split']
        self.connective_columns = ['word', 'count', 'type', 'temporal', 'flag']
        # loading spaCy's english model
        self.nlp = spacy.load("en_core_web_sm")
        self.semeval_2007_4_code = 1
        self.semeval_2010_8_code = 2
        self.event_causality_code = 3
        self.causal_timebank_code = 4
        self.storyline_code = 5
        self.caters_code = 6
        self.because_code = 7
        self.storyline_v15_code = 8

    def convert_semeval_2007_4(self):
        """
        reading SemEval 2007 task 4 data
        :return: pandas data frame of samples
        """

        data = []
        e1_tag = ["<e1>", "</e1>"]
        e2_tag = ["<e2>", "</e2>"]

        def extract_samples(all_lines, split):
            # each sample has three lines of information
            for idx, val in enumerate(all_lines):
                tmp = val.split(" ", 1)
                if tmp[0].isalnum():
                    sample_id = copy.deepcopy(tmp[0])
                    try:
                        context = copy.deepcopy(tmp[1].replace("\"", ""))

                        # extracting elements of causal relation
                        span1 = self._get_between_text(e1_tag[0], e1_tag[1], context)
                        span2 = self._get_between_text(e2_tag[0], e2_tag[1], context)

                        tmp = all_lines[idx + 1].split(",")
                        if not ("true" in tmp[3] or "false" in tmp[3]):
                            tmp_label = tmp[2].replace(" ", "").replace("\"", "").split("=")
                        else:
                            tmp_label = tmp[3].replace(" ", "").replace("\"", "").split("=")

                        # finding label
                        if tmp_label[1] == "true":
                            # if (e1, e2) = (cause, effect), label = 1,
                            # otherwise, label = 2 meaning (e2, e1) = (cause, effect)
                            if "e2" in tmp_label[0]:
                                label = 1
                            elif "e1" in tmp_label[0]:
                                label = 2
                        else:
                            label = 0

                        span1_start = context.find(e1_tag[0])
                        span1_end = context.find(e1_tag[1]) - len(e1_tag[0])
                        span2_start = context.find(e2_tag[0]) - (len(e1_tag[0]) + len(e1_tag[1]))
                        span2_end = context.find(e2_tag[1]) - (len(e1_tag[0]) + len(e1_tag[1]) + len(e2_tag[0]))

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]]}

                        # replacing tags with standard tags
                        context = context.replace(e1_tag[0], "").replace(e1_tag[1], "").replace(e2_tag[0], "").replace(
                            e2_tag[1], "")

                        data.append(
                            [int(sample_id), span1, span2, context, idx_val, label, self.semeval_2007_4_code, "",
                             split])
                    except Exception as e:
                        print("[crest-log] Incorrect formatting for semeval07-task4 record. Details: " + str(e))

        # reading files
        with open(self.dir_path + 'SemEval2007_task4/task-4-training/relation-1-train.txt', mode='r',
                  encoding='cp1252') as train:
            train_content = train.readlines()

        # this is the test set
        with open(self.dir_path + 'SemEval2007_task4/task-4-scoring/relation-1-score.txt', mode='r',
                  encoding='cp1252') as key:
            test_content = key.readlines()

        extract_samples(train_content, 0)
        extract_samples(test_content, 2)

        data = pd.DataFrame(data, columns=self.scheme_columns)

        assert self._check_span_indexes(data) == True

        return data

    def convert_semeval_2010_8(self):
        """
        reading SemEval 2010 task 8 data
        :return: pandas data frame of samples
        """
        data = []
        e1_tag = ["<e1>", "</e1>"]
        e2_tag = ["<e2>", "</e2>"]

        def extract_samples(all_lines, split):
            # each sample has three lines of information
            for idx, val in enumerate(all_lines):
                tmp = val.split("\t")
                if tmp[0].isalnum():
                    sample_id = copy.deepcopy(tmp[0])
                    try:
                        context = copy.deepcopy(tmp[1].replace("\"", ""))

                        # extracting elements of causal relation
                        span1 = self._get_between_text(e1_tag[0], e1_tag[1], context)
                        span2 = self._get_between_text(e2_tag[0], e2_tag[1], context)

                        # finding label
                        if "Cause-Effect" in all_lines[idx + 1]:
                            if "e1,e2" in all_lines[idx + 1]:
                                label = 1
                            else:
                                label = 2
                        else:
                            label = 0

                        span1_start = context.find(e1_tag[0])
                        span1_end = context.find(e1_tag[1]) - len(e1_tag[0])
                        span2_start = context.find(e2_tag[0]) - (len(e1_tag[0]) + len(e1_tag[1]))
                        span2_end = context.find(e2_tag[1]) - (len(e1_tag[0]) + len(e1_tag[1]) + len(e2_tag[0]))

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]]}

                        # replacing tags with standard tags
                        context = context.replace(e1_tag[0], "").replace(e1_tag[1], "").replace(e2_tag[0], "").replace(
                            e2_tag[1], "")

                        data.append(
                            [int(sample_id), span1, span2, context, idx_val, label, self.semeval_2010_8_code, "",
                             split])

                    except Exception as e:
                        print("[crest-log] Incorrect formatting for semeval10-task8 record. Details: " + str(e))

        # reading files
        with open(self.dir_path + 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.txt', mode='r',
                  encoding='cp1252') as train:
            train_content = train.readlines()

        with open(self.dir_path + 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt',
                  mode='r', encoding='cp1252') as key:
            test_content = key.readlines()

        extract_samples(train_content, 0)
        extract_samples(test_content, 2)

        data = pd.DataFrame(data, columns=self.scheme_columns)

        assert self._check_span_indexes(data) == True

        return data

    def convert_event_causality(self):

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
                # there're two types of tags -> C: causal, R: related (mixed)
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

        # reading all keys (annotations)
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

                        for child in root.findall("./P/S3"):
                            sen_id = child.attrib['id']
                            sentences[sen_id] = child.text
                        docs[doc_id] = sentences
                    except Exception as e:
                        print("[crest-log] Error in parsing XML file. Details: {}".format(e))

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

                    # if both spans are in the same sentence
                    if p1[0] == p2[0]:
                        context = ""
                        token_idx = 0

                        span1_token_idx = p1[1]
                        span2_token_idx = p2[1]
                        doc_sentences = docs[key]
                        tokens = doc_sentences[p1[0]].split(' ')

                        # finding spans 1 and 2 and building context
                        for i in range(len(tokens)):
                            token = tokens[i].split('/')[0]
                            if i == int(span1_token_idx):
                                span1 = copy.deepcopy(token)
                                span1_start = copy.deepcopy(token_idx)
                                span1_end = span1_start + len(span1)
                            elif i == int(span2_token_idx):
                                span2 = copy.deepcopy(token)
                                span2_start = copy.deepcopy(token_idx)
                                span2_end = span2_start + len(span2)
                            context += token + " "
                            token_idx += len(token) + 1

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]]}

                        data.append([data_idx, span1, span2, context, idx_val, 1, self.event_causality_code, "", split])

                        data_idx += 1
                    else:
                        # this means both spans are NOT in the same sentence
                        s_idx = {}
                        if int(p2[0]) < int(p1[0]):
                            s_idx[1] = {'sen_index': int(p2[0]), 'token_index': int(p2[1])}
                            s_idx[2] = {'sen_index': int(p1[0]), 'token_index': int(p1[1])}
                            label = 2
                        else:
                            s_idx[1] = {'sen_index': int(p1[0]), 'token_index': int(p1[1])}
                            s_idx[2] = {'sen_index': int(p2[0]), 'token_index': int(p2[1])}
                            label = 1

                        context = ""
                        token_idx = 0

                        doc_sentences = docs[key]
                        for k, v in doc_sentences.items():
                            tokens = v.split(' ')
                            if int(k) == s_idx[1]['sen_index']:
                                for i in range(len(tokens)):
                                    token = tokens[i].split('/')[0]

                                    # found spans1
                                    if i == int(s_idx[1]['token_index']):
                                        span1 = copy.deepcopy(token)
                                        span1_start = copy.deepcopy(token_idx)
                                        span1_end = span1_start + len(span1)

                                    context += token + " "
                                    token_idx += len(token) + 1
                            elif int(k) == s_idx[2]['sen_index']:
                                for i in range(len(tokens)):
                                    token = tokens[i].split('/')[0]

                                    # found span2
                                    if i == int(s_idx[2]['token_index']):
                                        span2 = copy.deepcopy(token)
                                        span2_start = copy.deepcopy(token_idx)
                                        span2_end = span2_start + len(span2)

                                    context += token + " "
                                    token_idx += len(token) + 1
                                break
                            elif s_idx[1]['sen_index'] < int(k) < s_idx[2]['sen_index']:
                                next_sen = get_sentence_text(v)
                                context += next_sen + " "
                                token_idx += len(next_sen) + 1

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]]}

                        data.append(
                            [data_idx, span1, span2, context, idx_val, label, self.event_causality_code, "", split])
                        data_idx += 1

        data = pd.DataFrame(data, columns=self.scheme_columns)

        assert self._check_span_indexes(data) == True

        return data

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
                                sentence += tokens[i + l][1] + " "
                                e1_tokens += tokens[i + l][1] + " "
                            sentence = sentence.strip() + self.arg1_tag[1] + " "

                        elif token_id == end_anchors[0]:
                            sentence = sentence + self.arg2_tag[0]
                            for l in range(len(end_anchors)):
                                sentence += tokens[i + l][1] + " "
                                e2_tokens += tokens[i + l][1] + " "
                            sentence = sentence.strip() + self.arg2_tag[1] + " "

                        elif token_id == signal_anchors[0]:
                            sentence += self.signal_tag[0]
                            for l in range(len(signal_anchors)):
                                sentence += tokens[i + l][1] + " "
                            sentence = sentence.strip() + self.signal_tag[1] + " "
                        else:
                            sentence += token_text + " "

                if self._check_text_format(e1_tokens, e2_tokens, sentence):
                    data.append([data_idx, e1_tokens, e2_tokens, sentence.replace('\n', ' '), label_direction, 1,
                                 self.causal_timebank_code, file, ""])
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
                        data.append([data_idx, e1_tokens, e2_tokens, sentence.replace('\n', ' '), label_direction, 1,
                                     self.storyline_code, "", ""])
                        data_idx += 1
                except Exception as e:
                    print(str(e), doc_id, sentence_id)
                    err += 1
        # print("Total samples = " + str(len(data)))
        if err > 0:
            print("[datareader-story-line-log] err: " + str(err))
        return pd.DataFrame(data, columns=self.scheme_columns)

    def read_story_lines_v_1_5(self):
        """
        reading causal and non-causal samples from EventStoryLines v1.5
        """

        def get_tagged_sentence(tokens, sentences, markables, relation):
            """
            getting arguments text spans and tagged sentence
            :param tokens: tokens in document
            :param sentences: sentences in document
            :param markables: tagged events in document
            :param relation: causal and non-causal relation info
            :return:
            """
            source_m_id = relation['source']
            target_m_id = relation['target']

            sentence_text = ""
            source_text = ""
            target_text = ""
            sent_ids = set()
            for item in markables[source_m_id]:
                source_text += tokens[item.attrib['t_id']]['text'] + " "
                sent_ids.add(tokens[item.attrib['t_id']]['s_id'])
            for item in markables[target_m_id]:
                target_text += tokens[item.attrib['t_id']]['text'] + " "
                sent_ids.add(tokens[item.attrib['t_id']]['s_id'])
            for sen_id in sent_ids:
                sentence_text += sentences[sen_id]

            arg1 = source_text.replace('\n', ' ').strip()
            arg2 = target_text.replace('\n', ' ').strip()

            return arg1, arg2, self.get_tagged_sentence(arg1, arg2, sentence_text).strip()

        docs_path = self.dir_path + "EventStoryLine/annotated_data/v1.5"

        # creating a dictionary of all documents
        data_idx = 1
        data = []
        err = 0
        for folder in os.listdir(docs_path):
            if not any(sub in folder for sub in [".txt", ".DS_Store"]):
                for doc in os.listdir(docs_path + "/" + folder):
                    if ".xml" in doc:
                        # initialization
                        doc_sentences = {}
                        relations_info = {}
                        markables_info = {}
                        tokens_info = {}

                        # parse the doc to retrieve info of sentences
                        tree = ET.parse(docs_path + "/" + folder + "/" + doc)
                        root = tree.getroot()
                        current_sentence = ""
                        current_sentence_id = -1

                        for token in root.findall('token'):
                            # saving token info
                            tokens_info[token.attrib['t_id']] = {'text': token.text, 's_id': token.attrib['sentence']}

                            if token.attrib['sentence'] != current_sentence_id and current_sentence_id != -1:
                                doc_sentences[current_sentence_id] = current_sentence.strip()
                                current_sentence = ""
                            current_sentence_id = token.attrib['sentence']
                            current_sentence += token.text + " "
                        # saving the last sentence
                        doc_sentences[current_sentence_id] = current_sentence.strip()

                        # saving relations info
                        rel_idx = 0
                        for relation in root.findall("Relations/PLOT_LINK"):
                            if all(rel_type in relation.attrib for rel_type in ['CAUSES', 'CAUSED_BY']) and \
                                    relation.attrib['validated'] == "TRUE":
                                if relation.attrib['CAUSES'] == "TRUE":
                                    label = 1
                                    direction = 0
                                elif relation.attrib['CAUSED_BY'] == "TRUE":
                                    label = 1
                                    direction = 1
                                else:
                                    label = 0
                                    direction = 0

                                relations_info[rel_idx] = {'source': relation[0].attrib['m_id'],
                                                           'target': relation[1].attrib['m_id'], 'label': label,
                                                           'direction': direction}
                                rel_idx += 1

                        # saving markables info
                        for markable in root.findall("Markables/"):
                            markables_info[markable.attrib['m_id']] = markable

                        # storing causal and non-causal info
                        try:
                            for k, v in relations_info.items():
                                e1_tokens, e2_tokens, sentence = get_tagged_sentence(tokens_info, doc_sentences,
                                                                                     markables_info, v)
                                if sentence != "":
                                    data.append(
                                        [data_idx, e1_tokens, e2_tokens, sentence.replace('\n', ' '), v['direction'],
                                         v['label'], self.storyline_v15_code, "", ""])
                                    data_idx += 1
                        except Exception as e:
                            print(str(e))
                            err += 1
        if err > 0:
            print("[datareader-story-line-v1.5-log] err: " + str(err))
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
                                            data.append(
                                                [data_idx, e1_tokens, e2_tokens, sentence.replace('\n', ' '), 0, 1,
                                                 self.caters_code, doc, split])
                                            data_idx += 1
                                except Exception as e:
                                    print(
                                        "[datareader-caters-log] Error in processing causal relation info. Details: " + str(
                                            e))
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

            for i in range(len(sentences) - 1, -1, -1):
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

            doc_cleaned = (" ".join(sentences[s_index:e_index + 1])).replace(' >', '>')

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
                            if (
                                    "Consequence" in value or "Purpose" in value or "Motivation" in value) and "Effect" in value and "Cause" in value:
                                args = value.split(' ')
                                signal_tag = args[0].split(':')[1]
                                if "Cause" in args[1]:
                                    cause_tag = args[1].split(':')[1]
                                    effect_tag = args[2].split(':')[1]
                                else:
                                    cause_tag = args[2].split(':')[1]
                                    effect_tag = args[1].split(':')[1]

                                e1_tokens, e2_tokens, doc_text = _process_args(doc_string, cause_tag, effect_tag,
                                                                               signal_tag)

                                if self._check_text_format(e1_tokens, e2_tokens, _remove_extra_sentences(doc_text)):
                                    data.append([data_idx, e1_tokens, e2_tokens,
                                                 _remove_extra_sentences(doc_text).replace('\n', ' '), 0, 1,
                                                 self.because_code, doc, ""])
                                    data_idx += 1

                            # non-causal samples
                            elif ("NonCausal" in value) and "Arg0" in value and "Arg1" in value:
                                args = value.split(' ')
                                signal_tag = args[0].split(':')[1]
                                arg0_tag = args[1].split(':')[1]
                                arg1_tag = args[2].split(':')[1]
                                e1_tokens, e2_tokens, doc_text = _process_args(doc_string, arg0_tag, arg1_tag,
                                                                               signal_tag)

                                if self._check_text_format(e1_tokens, e2_tokens, _remove_extra_sentences(doc_text)):
                                    data.append([data_idx, e1_tokens, e2_tokens,
                                                 _remove_extra_sentences(doc_text).replace('\n', ' '), 0, 0,
                                                 self.because_code, doc, ""])
                                    data_idx += 1

                        except Exception as e:
                            print(
                                "[datareader-because-log] Error in processing causal relation info. Details: " + str(e))
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

    def remove_text_tags(self, text):
        text = text.replace(self.signal_tag[0], "")
        text = text.replace(self.signal_tag[1], "")
        text = text.replace(self.arg1_tag[0], "")
        text = text.replace(self.arg1_tag[1], "")
        text = text.replace(self.arg2_tag[0], "")
        text = text.replace(self.arg2_tag[1], "")
        return text.strip().lower()

    def get_text_between(self, x, arg1_tag=[], arg2_tag=[]):
        """
        this method gets the text between two arguments of a causal relation
        :param x: the text field of a record with tags in the causal relation data frame
        :param arg1_tag: by default, the first tag in our causal scheme
        :param arg2_tag: by default, the second tag in our causal schema
        :return:
        """

        def _clean_text(text):
            # TODO: ideally we should only be removing signal tags.
            text = text.replace(self.signal_tag[0], "")
            text = text.replace(self.signal_tag[1], "")
            text = text.replace(self.arg1_tag[0], "")
            text = text.replace(self.arg1_tag[1], "")
            text = text.replace(self.arg2_tag[0], "")
            text = text.replace(self.arg2_tag[1], "")
            return text

        if arg1_tag == [] and arg2_tag == []:
            arg1_tag = self.arg1_tag
            arg2_tag = self.arg2_tag

        try:
            # in cases that we have <arg1></arg1> . . . <arg2></arg2>
            result = re.search(arg1_tag[1] + "(.*)" + arg2_tag[0], x)
            if result is None:
                # in cases that we have <arg2></arg2> . . . <arg1></arg1>
                result = re.search(arg2_tag[1] + "(.*)" + arg1_tag[0], x)
                if result is None:
                    return ""
                else:
                    return _clean_text(result.group(1))
            else:
                return _clean_text(result.group(1))
        except Exception as e:
            print("[log-preprocessing-text-between] detail: " + str(e) + ". text: " + str(x))

    def get_text_left(self, x):
        """
        this method gets the text of the left side of the first argument in a causal relation
        :param x: the text field of a record with tags in the causal relation data frame
        :return:
        """

        arg1_tag = self.arg1_tag
        arg2_tag = self.arg2_tag

        # in cases that we have <arg1></arg1> . . . <arg2></arg2>
        result = re.search(arg1_tag[1] + "(.*)" + arg2_tag[0], x)
        if result is None:
            # in cases that we have <arg2></arg2> . . . <arg1></arg1>
            result = re.search(arg2_tag[1] + "(.*)" + arg1_tag[0], x)
            if result is not None:
                result = re.search("(.*)" + arg2_tag[0], x)
                if result is not None:
                    return result.group(1)
        else:
            result = re.search("(.*)" + arg1_tag[0], x)
            if result is not None:
                return result.group(1)
        return ""

    def get_text_right(self, x):
        """
        this method gets the text of the left side of the first argument in a causal relation
        :param x: the text field of a record with tags in the causal relation data frame
        :return:
        """

        arg1_tag = self.arg1_tag
        arg2_tag = self.arg2_tag

        # in cases that we have <arg1></arg1> . . . <arg2></arg2>
        result = re.search(arg1_tag[1] + "(.*)" + arg2_tag[0], x)
        if result is None:
            # in cases that we have <arg2></arg2> . . . <arg1></arg1>
            result = re.search(arg2_tag[1] + "(.*)" + arg1_tag[0], x)
            if result is not None:
                result = re.search(arg1_tag[1] + "(.*)", x)
                if result is not None:
                    return result.group(1)
        else:
            result = re.search(arg2_tag[1] + "(.*)", x)
            if result is not None:
                return result.group(1)
        return ""

    def get_tagged_sentence(self, arg1, arg2, sentence):
        """
        adding arguments tags to the sentence in the format of <arg1></arg1> . . . <arg2></arg2>
        :return:
        """
        tagged_sen = ""
        try:
            # for simplicity, check if there's only one occurrence of each argument in the sentence
            if sentence.count(arg1) == 1 and sentence.count(arg2) == 1:
                # making sure <arg1> is always first
                if sentence.index(arg1) < sentence.index(arg2):
                    tagged_sen = sentence.replace(arg1, self.arg1_tag[0] + arg1 + self.arg1_tag[1])
                    tagged_sen = tagged_sen.replace(arg2, self.arg2_tag[0] + arg2 + self.arg2_tag[1])
                else:
                    tagged_sen = sentence.replace(arg2, self.arg1_tag[0] + arg2 + self.arg1_tag[1])
                    tagged_sen = tagged_sen.replace(arg1, self.arg2_tag[0] + arg1 + self.arg2_tag[1])
        except Exception as e:
            print("[log-sentence-tagging]: " + str(e))
        return tagged_sen

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
                all_sen.append((row['start_tag'] + doc_text[row['start']:row['end']] + row['end_tag'] + doc_text[
                                                                                                        row['end']:
                                                                                                        tags.iloc[
                                                                                                            index + 1][
                                                                                                            'start']]).strip())

        all_sen.append((tags.iloc[len(tags) - 1]['start_tag'] + doc_text[tags.iloc[len(tags) - 1]['start']:
                                                                         tags.iloc[len(tags) - 1]['end']] +
                        tags.iloc[len(tags) - 1]['end_tag'] + doc_text[tags.iloc[len(tags) - 1]['end']:]).strip())

        sentence = ' '.join(s for s in all_sen).strip()

        return sentence

    @staticmethod
    def _check_span_indexes(data):
        """
        checking if span indexes are correctly stored
        :param data:
        :return:
        """
        for index, row in data.iterrows():
            span1 = row["context"][row["idx"]["span1"][0][0]:row["idx"]["span1"][0][1]]
            span2 = row["context"][row["idx"]["span2"][0][0]:row["idx"]["span2"][0][1]]
            if span1 != row["span1"] or span2 != row["span2"]:
                return False
            return True

    @staticmethod
    def _get_tag_text(tag, text):
        """
        get text span of a tag (if there's multiple occurrences, we return the first occurrence)
        :param tag: a list of opening and closing tags
        :param text: tagged sentence/text
        :return:
        """
        result = re.findall(tag[0] + "(.*?)" + tag[1], text)
        if result is not None and len(result) > 0:
            return result[0].strip()
        return ""
