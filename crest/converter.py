import re
import os
import sys
import copy
import spacy
import logging
import pandas as pd
import xml.etree.ElementTree as ET


class Converter:
    """
    idx = {'span1': [], 'span2': [], 'signal': []} -> indexes of span1, span2, and signal tokens/spans in context
    each value in the idx dictionary is a list of lists of indexes. For example, if span1 has multi tokens in context
    with start:end indexes 2:5 and 10:13, respectively, span1's value in 'idx' will be [[2, 5],[10, 13]]. Lists are
    sorted based on the start indexes of tokens. Same applies for span2 and signal.
    -------------------------------------------------------------------------------
    label => 0: non-causal, 1: causal
    direction => 0: span1 => span2, 1: span2 => span1, -1: not-specified
    -------------------------------------------------------------------------------
    split -> 0: train, 1: dev, test: 2. This is the split/part that a relation belongs to in the original dataset.
    For example, if split value for a relation is 1, it means that in the original dataset, the relation is used in the
    development set
    """

    def __init__(self):
        root_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
        sys.path.insert(0, root_path)
        self.dir_path = root_path + "/data/"
        self.scheme_columns = ['original_id', 'span1', 'span2', 'signal', 'context', 'idx', 'label', 'direction',
                               'source', 'ann_file', 'split']

        # loading spaCy's english model (we use spaCy's sentence splitter for long context)
        self.nlp = spacy.load("en_core_web_sm")

        self.namexid = {"semeval_2007_4": 1,
                        "semeval_2010_8": 2,
                        "event_causality": 3,
                        "causal_timebank": 4,
                        "eventstorylines": 5,
                        "caters": 6,
                        "because": 7,
                        "copa": 8,
                        "pdtb3": 9,
                        }

        self.idxmethod = {self.namexid["semeval_2007_4"]: self.convert_semeval_2007_4,
                          self.namexid["semeval_2010_8"]: self.convert_semeval_2010_8,
                          self.namexid["event_causality"]: self.convert_event_causality,
                          self.namexid["causal_timebank"]: self.convert_causal_timebank,
                          self.namexid["eventstorylines"]: self.convert_eventstorylines_v1,
                          self.namexid["caters"]: self.convert_caters,
                          self.namexid["because"]: self.convert_because,
                          self.namexid["copa"]: self.convert_copa,
                          self.namexid["pdtb3"]: self.convert_pdtb3
                          }

    def convert2crest(self, dataset_ids=[], save_file=False):
        """
        converting a dataset to CREST
        :param dataset_ids: list of integer ids of datasets
        :param save_file: True if want saving result dataframe into xml, False, otherwise
        :return:
        """
        data = pd.DataFrame(columns=self.scheme_columns)
        total_mis = 0
        for key, value in self.idxmethod.items():
            if key in dataset_ids:
                df, mis = value()
                data = data.append(df).reset_index(drop=True)
                total_mis += mis
        if save_file:
            data.to_excel(self.dir_path + "crest.xlsx")

        return data, total_mis

    def convert_semeval_2007_4(self):
        """
        reading SemEval 2007 task 4 data
        :return: pandas data frame of samples
        """
        e1_tag = ["<e1>", "</e1>"]
        e2_tag = ["<e2>", "</e2>"]
        global mismatch
        mismatch = 0

        def extract_samples(all_lines, split):
            samples = pd.DataFrame(columns=self.scheme_columns)
            # each sample has three lines of information
            for idx, val in enumerate(all_lines):
                tmp = val.split(" ", 1)
                if tmp[0].isalnum():
                    original_id = copy.deepcopy(tmp[0])
                    try:
                        context = copy.deepcopy(tmp[1].replace("\"", ""))

                        # extracting elements of causal relation
                        span1 = self._get_between_text(e1_tag[0], e1_tag[1], context)
                        span2 = self._get_between_text(e2_tag[0], e2_tag[1], context)

                        tmp = all_lines[idx + 1].split(",")
                        if not ("true" in tmp[3] or "false" in tmp[3]):
                            tmp_label = tmp[2].replace(" ", "").replace("\"", "").split("=")
                            relation_type = tmp[1]
                        else:
                            tmp_label = tmp[3].replace(" ", "").replace("\"", "").split("=")
                            relation_type = tmp[2]

                        # finding label
                        if "Cause-Effect" in relation_type and tmp_label[1] == "true":
                            label = 1
                        else:
                            label = 0

                        # finding direction
                        # if 0: e1 => e2, if 1: e2 => e1
                        if "e2" in tmp_label[0]:
                            direction = 0
                        elif "e1" in tmp_label[0]:
                            direction = 1

                        span1_start = context.find(e1_tag[0])
                        span1_end = context.find(e1_tag[1]) - len(e1_tag[0])
                        span2_start = context.find(e2_tag[0]) - (len(e1_tag[0]) + len(e1_tag[1]))
                        span2_end = context.find(e2_tag[1]) - (len(e1_tag[0]) + len(e1_tag[1]) + len(e2_tag[0]))

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                   "signal": []}

                        # replacing tags with standard tags
                        context = context.replace(e1_tag[0], "").replace(e1_tag[1], "").replace(e2_tag[0], "").replace(
                            e2_tag[1], "")

                        new_row = {"original_id": int(original_id), "span1": [span1], "span2": [span2], "signal": [],
                                   "context": context.strip('\n'),
                                   "idx": idx_val, "label": label, "direction": direction,
                                   "source": self.namexid["semeval_2007_4"],
                                   "ann_file": "",
                                   "split": split}

                        # span1_end < span2_start is to make sure e1 always appears first
                        # in context and direction is correct
                        if self._check_span_indexes(new_row) and span1_end < span2_start:
                            samples = samples.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1

                    except Exception as e:
                        print("[crest-log] semeval07-task4. Detail: {}".format(e))
            return samples

        data = pd.DataFrame(columns=self.scheme_columns)

        relation_ids = [1, 2, 3, 4, 5, 6, 7]

        for relation_id in relation_ids:
            # reading files
            with open(
                    self.dir_path + 'SemEval2007_task4/task-4-training/relation-{}-train.txt'.format(str(relation_id)),
                    mode='r',
                    encoding='cp1252') as train:
                train_content = train.readlines()

            # this is the test set
            with open(self.dir_path + 'SemEval2007_task4/task-4-scoring/relation-{}-score.txt'.format(str(relation_id)),
                      mode='r',
                      encoding='cp1252') as key:
                test_content = key.readlines()

            data = data.append(extract_samples(train_content, 0))
            data = data.append(extract_samples(test_content, 2))

        logging.info("[crest] semeval_2007_4 is converted.")

        return data, mismatch

    def convert_semeval_2010_8(self):
        """
        reading SemEval 2010 task 8 data
        :return: pandas data frame of samples
        """
        e1_tag = ["<e1>", "</e1>"]
        e2_tag = ["<e2>", "</e2>"]
        global mismatch
        mismatch = 0

        def extract_samples(all_lines, split):
            samples = pd.DataFrame(columns=self.scheme_columns)
            # each sample has three lines of information
            for idx, val in enumerate(all_lines):
                tmp = val.split("\t")
                if tmp[0].isalnum():
                    original_id = copy.deepcopy(tmp[0])
                    try:
                        context = copy.deepcopy(tmp[1].replace("\"", ""))

                        # extracting elements of causal relation
                        span1 = self._get_between_text(e1_tag[0], e1_tag[1], context)
                        span2 = self._get_between_text(e2_tag[0], e2_tag[1], context)

                        # finding label
                        if "Cause-Effect" in all_lines[idx + 1]:
                            label = 1
                        else:
                            label = 0

                        # finding direction
                        if "e1,e2" in all_lines[idx + 1]:
                            direction = 0
                        elif "e2,e1" in all_lines[idx + 1]:
                            direction = 1
                        else:
                            direction = -1

                        span1_start = context.find(e1_tag[0])
                        span1_end = context.find(e1_tag[1]) - len(e1_tag[0])
                        span2_start = context.find(e2_tag[0]) - (len(e1_tag[0]) + len(e1_tag[1]))
                        span2_end = context.find(e2_tag[1]) - (len(e1_tag[0]) + len(e1_tag[1]) + len(e2_tag[0]))

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                   "signal": []}

                        # replacing tags with standard tags
                        context = context.replace(e1_tag[0], "").replace(e1_tag[1], "").replace(e2_tag[0],
                                                                                                "").replace(
                            e2_tag[1], "")

                        new_row = {"original_id": int(original_id), "span1": [span1], "span2": [span2],
                                   "signal": [],
                                   "context": context.strip('\n'), "idx": idx_val, "label": label,
                                   "direction": direction,
                                   "source": self.namexid["semeval_2010_8"], "ann_file": "", "split": split}

                        if self._check_span_indexes(new_row) and span1_end < span2_start:
                            samples = samples.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1

                    except Exception as e:
                        print("[crest-log] Incorrect formatting for semeval10-task8 record. Detail: " + str(e))
            return samples

        # reading files
        with open(self.dir_path + 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.txt', mode='r',
                  encoding='cp1252') as train:
            train_content = train.readlines()

        with open(self.dir_path + 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt',
                  mode='r', encoding='cp1252') as key:
            test_content = key.readlines()

        data = pd.DataFrame(columns=self.scheme_columns)

        data = data.append(extract_samples(train_content, 0))
        data = data.append(extract_samples(test_content, 2))

        logging.info("[crest] semeval_2010_8 is converted.")

        return data, mismatch

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
                            orig_id = tag.replace('\t', ' ').replace(' ', '_') + str(doc_id)
                            # the first combo is always CAUSE and the second combo is EFFECT
                            doc_tags.append({'p1': tag_var[1], 'p2': tag_var[2], 'split': split,
                                             'original_id': orig_id})
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

        data = pd.DataFrame(columns=self.scheme_columns)

        mismatch = 0

        # now that we have all the information in dictionaries, we create samples
        for key, values in keys.items():
            if key in docs:
                # each key is a doc id
                for value in values:
                    original_id = value['original_id']
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

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                   "signal": []}

                        new_row = {"original_id": original_id, "span1": [span1], "span2": [span2], "signal": [],
                                   "context": context.strip('\n'),
                                   "idx": idx_val, "label": 1, "direction": 0,
                                   "source": self.namexid["event_causality"],
                                   "ann_file": key,
                                   "split": split}

                        if self._check_span_indexes(new_row):
                            data = data.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1
                    else:
                        # this means both spans are NOT in the same sentence
                        s_idx = {}
                        if int(p2[0]) < int(p1[0]):
                            s_idx[1] = {'sen_index': int(p2[0]), 'token_index': int(p2[1])}
                            s_idx[2] = {'sen_index': int(p1[0]), 'token_index': int(p1[1])}
                            direction = 1
                        else:
                            s_idx[1] = {'sen_index': int(p1[0]), 'token_index': int(p1[1])}
                            s_idx[2] = {'sen_index': int(p2[0]), 'token_index': int(p2[1])}
                            direction = 0

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

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                   "signal": []}

                        new_row = {"original_id": original_id, "span1": [span1], "span2": [span2], "signal": [],
                                   "context": context.strip('\n'),
                                   "idx": idx_val, "label": 1, "direction": direction,
                                   "source": self.namexid["event_causality"],
                                   "ann_file": key,
                                   "split": split}

                        if self._check_span_indexes(new_row):
                            data = data.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1

        logging.info("[crest] event_causality is converted.")

        return data, mismatch

    def convert_causal_timebank(self):
        """
        converting samples from Causal-TimeBank
        """
        mismatch = 0
        data_path = self.dir_path + "Causal-TimeBank/Causal-TimeBank-CAT"
        all_files = os.listdir(data_path)
        # parser = ET.XMLParser(encoding="utf-8")
        data = pd.DataFrame(columns=self.scheme_columns)
        for file in all_files:
            tokens = []
            try:
                tree = ET.parse(data_path + "/" + file)
                root = tree.getroot()
            except Exception as e:
                print("file: {}, error: {}".format(file, e))

            # [0] getting information of events
            events = {}
            markables = ['Markables/EVENT', 'Markables/TIMEX3', 'Markables/C-SIGNAL']
            for markable in markables:
                for event in root.findall(markable):
                    events_ids = []
                    for anchor in event:
                        events_ids.append(int(anchor.attrib['id']))
                    events[int(event.attrib['id'])] = events_ids

            # [1] getting list of tokens in sentence/doc
            for token in root.findall('token'):
                tokens.append(
                    [int(token.attrib['id']), token.text, int(token.attrib['number']), int(token.attrib['sentence'])])

            # [2] getting list of causal links
            for link in root.findall('Relations/CLINK'):
                original_id = link.attrib['id']
                s_t = {link[0].tag: link[0], link[1].tag: link[1]}
                s_event_id = int(s_t['source'].attrib['id'])  # source event id
                t_event_id = int(s_t['target'].attrib['id'])  # target event id

                if 'c-signalID' in link.attrib:
                    signal_id = int(link.attrib['c-signalID'])
                else:
                    signal_id = False

                context = ""
                span1 = ""
                span2 = ""
                signal = ""
                direction = 0
                token_idx = 0

                # finding start and end sentences indexes
                for i in range(len(tokens)):
                    if tokens[i][0] == events[s_event_id][0]:
                        s_sen_id = int(tokens[i][3])
                    if tokens[i][0] == events[t_event_id][0]:
                        t_sen_id = int(tokens[i][3])

                if s_sen_id > t_sen_id:
                    s_sen_id, t_sen_id = t_sen_id, s_sen_id

                # building the context and finding spans
                i = 0
                while i < len(tokens):
                    token_id = int(tokens[i][0])
                    token_text = tokens[i][1]
                    token_sen_id = int(tokens[i][3])
                    if s_sen_id <= int(token_sen_id) <= t_sen_id:
                        # span1
                        if token_id == events[s_event_id][0]:
                            for l in range(len(events[s_event_id])):
                                span1 += tokens[i + l][1] + " "
                            # setting span1 start and end indexes
                            span1_start = copy.deepcopy(token_idx)
                            span1_end = span1_start + len(span1) - 1
                            context += span1
                            token_idx += len(span1)
                            i += l

                        # span2
                        elif token_id == events[t_event_id][0]:
                            for l in range(len(events[t_event_id])):
                                span2 += tokens[i + l][1] + " "
                            # setting span2 start and end indexes
                            span2_start = copy.deepcopy(token_idx)
                            span2_end = span2_start + len(span2) - 1
                            context += span2
                            token_idx += len(span2)
                            i += l

                        # signal token
                        elif signal_id and token_id == events[signal_id][0]:
                            for l in range(len(events[signal_id])):
                                signal += tokens[i + l][1] + " "
                            # setting signal start and end indexes
                            signal_start = copy.deepcopy(token_idx)
                            signal_end = signal_start + len(signal) - 1
                            context += signal
                            token_idx += len(signal)
                            i += l
                        else:
                            context += token_text + " "
                            token_idx += len(token_text) + 1
                    i += 1

                idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]], "signal": []}

                if signal.strip() != "":
                    idx_val["signal"].append([signal_start, signal_end])

                new_row = {"original_id": original_id, "span1": [span1.strip()], "span2": [span2.strip()],
                           "signal": [signal.strip()],
                           "context": context.strip('\n'), "idx": idx_val, "label": 1, "direction": direction,
                           "source": self.namexid["causal_timebank"],
                           "ann_file": file, "split": ""}

                if self._check_span_indexes(new_row):
                    data = data.append(new_row, ignore_index=True)
                else:
                    mismatch += 1

        logging.info("[crest] causal_timebank is converted.")

        return data, mismatch

    def convert_eventstorylines_v1(self, version="1.5"):
        """
        converting causal and non-causal samples from EventStoryLines
        """
        mismatch = 0
        docs_path = self.dir_path + "EventStoryLine/annotated_data/v" + version

        # creating a dictionary of all documents
        data = pd.DataFrame(columns=self.scheme_columns)

        for folder in os.listdir(docs_path):
            if not any(sub in folder for sub in [".txt", ".pdf", ".DS_Store"]):
                for doc in os.listdir(docs_path + "/" + folder):
                    if ".xml" in doc:
                        # initialization
                        markables = {}
                        tokens = []

                        # parse the doc to retrieve info of sentences
                        tree = ET.parse(docs_path + "/" + folder + "/" + doc)
                        root = tree.getroot()

                        # saving tokens info
                        for token in root.findall('token'):
                            tokens.append([int(token.attrib['t_id']), token.text, int(token.attrib['sentence'])])

                        # saving markables info
                        for markable in root.findall("Markables/"):
                            anchor_ids = []
                            for anchor in markable:
                                anchor_ids.append(int(anchor.attrib['t_id']))
                            markables[int(markable.attrib['m_id'])] = anchor_ids

                        # saving relations info
                        # "CAUSES" and "CAUSED_BY" are for marking explicit causal relations
                        for relation in root.findall("Relations/PLOT_LINK"):
                            label = 0
                            direction = 0
                            if "relType" in relation.attrib:
                                if relation.attrib['relType'] == "PRECONDITION":
                                    label = 1
                                elif relation.attrib['relType'] == "FALLING_ACTION":
                                    label = 1
                                    direction = 1

                                # --------------------------
                                # building context and spans
                                original_id = relation.attrib["r_id"]
                                source_target = {relation[0].tag: relation[0], relation[1].tag: relation[1]}
                                source_m_id = int(source_target['source'].attrib['m_id'])
                                target_m_id = int(source_target['target'].attrib['m_id'])

                                context = ""
                                span1 = ""
                                span2 = ""
                                token_idx = 0

                                # finding start and end sentences indexes
                                for i in range(len(tokens)):
                                    if tokens[i][0] == markables[source_m_id][0]:
                                        s_sen_id = int(tokens[i][2])
                                    if tokens[i][0] == markables[target_m_id][0]:
                                        t_sen_id = int(tokens[i][2])

                                # building the context and finding spans
                                i = 0

                                if t_sen_id < s_sen_id:
                                    s_sen_id, t_sen_id = t_sen_id, s_sen_id

                                while i < len(tokens):
                                    t_id = tokens[i][0]
                                    token_text = tokens[i][1]
                                    token_sen_id = int(tokens[i][2])
                                    if s_sen_id <= int(token_sen_id) <= t_sen_id:
                                        # span1
                                        if t_id == markables[source_m_id][0]:
                                            for l in range(len(markables[source_m_id])):
                                                span1 += tokens[i + l][1] + " "
                                            # setting span1 start and end indexes
                                            span1_start = copy.deepcopy(token_idx)
                                            span1_end = span1_start + len(span1) - 1
                                            context += span1
                                            token_idx += len(span1)
                                            i += l

                                        # span2
                                        elif t_id == markables[target_m_id][0]:
                                            for l in range(len(markables[target_m_id])):
                                                span2 += tokens[i + l][1] + " "
                                            # setting span2 start and end indexes
                                            span2_start = copy.deepcopy(token_idx)
                                            span2_end = span2_start + len(span2) - 1
                                            context += span2
                                            token_idx += len(span2)
                                            i += l
                                        else:
                                            context += token_text + " "
                                            token_idx += len(token_text) + 1
                                    i += 1
                                # --------------------------

                                # storing causal and non-causal info
                                try:
                                    idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                               "signal": []}

                                    new_row = {"original_id": original_id, "span1": [span1.strip()],
                                               "span2": [span2.strip()],
                                               "signal": [],
                                               "context": context.strip('\n'), "idx": idx_val, "label": label,
                                               "direction": direction,
                                               "source": self.namexid["eventstorylines"],
                                               "ann_file": doc, "split": ""}

                                    if self._check_span_indexes(new_row):
                                        data = data.append(new_row, ignore_index=True)
                                    else:
                                        mismatch += 1

                                except Exception as e:
                                    print("[crest-log] EventStoryLine. Detail: {}".format(e))

        logging.info("[crest] eventstorylines is converted.")

        return data, mismatch

    def convert_eventstorylines_v2(self, version="1.5"):
        """
        converting causal and non-causal samples from EventStoryLines based on evaluation_format file
        """

        splits = {'full_corpus/v{}/event_mentions_extended'.format(version): 0,
                  'test_corpus/v{}/event_mentions_extended'.format(version): 2}

        annotations = pd.DataFrame(columns=['file', 'source', 'target', 'label', 'split'])

        # creating a dictionary of all documents
        data = pd.DataFrame(columns=self.scheme_columns)

        global_id = 0

        # ----------------------------------
        # reading all the annotations
        for key, value in splits.items():
            docs_path = self.dir_path + "EventStoryLine/evaluation_format/{}".format(key)

            for folder in os.listdir(docs_path):
                if not any(sub in folder for sub in [".txt", ".pdf", ".DS_Store"]):
                    for doc in os.listdir('{}/{}'.format(docs_path, folder)):
                        if ".tab" in doc:
                            with open('{}/{}/{}'.format(docs_path, folder, doc), 'r') as file:
                                lines = file.readlines()
                            for line in lines:
                                line = line.split('\t')
                                annotations = annotations.append(
                                    {'file': '{}.{}'.format(doc.split('.')[0], 'xml'), 'source': line[0],
                                     'target': line[1],
                                     'label': line[2].replace('\n', ''), 'split': value}, ignore_index=True)

        # ----------------------------------
        mismatch = 0
        docs_path = self.dir_path + "EventStoryLine/ECB+_LREC2014/ECB+"

        # creating a dictionary of all documents
        data = pd.DataFrame(columns=self.scheme_columns)

        for index, row in annotations.iterrows():
            # parse the doc to retrieve info of sentences
            folder = str(row['file']).split('_')[0]
            tree = ET.parse(docs_path + "/" + folder + "/" + row['file'])
            root = tree.getroot()

            tokens = []

            # saving tokens info
            for token in root.findall('token'):
                tokens.append([int(token.attrib['t_id']), token.text, int(token.attrib['sentence'])])

            label = -1
            direction = -1
            if str(row['label']) == "PRECONDITION":
                label = 1
                direction = 0
            elif str(row['label']) == "FALLING_ACTION":
                label = 1
                direction = 1

            source_t_ids = []
            target_t_ids = []
            for item in row['source'].split('_'):
                source_t_ids.append(int(item))
            for item in row['target'].split('_'):
                target_t_ids.append(int(item))

            context = ""
            span1 = ""
            span2 = ""
            token_idx = 0

            # finding start and end sentences indexes
            for i in range(len(tokens)):
                if tokens[i][0] == source_t_ids[0]:
                    s_sen_id = int(tokens[i][2])
                if tokens[i][0] == target_t_ids[-1]:
                    t_sen_id = int(tokens[i][2])

            # building the context and finding spans
            i = 0

            if t_sen_id < s_sen_id:
                s_sen_id, t_sen_id = t_sen_id, s_sen_id

            while i < len(tokens):
                t_id = tokens[i][0]
                token_text = tokens[i][1]
                token_sen_id = int(tokens[i][2])
                if s_sen_id <= int(token_sen_id) <= t_sen_id:
                    # span1
                    if t_id == source_t_ids[0]:
                        for l in range(len(source_t_ids)):
                            span1 += tokens[i + l][1] + " "
                        # setting span1 start and end indexes
                        span1_start = copy.deepcopy(token_idx)
                        span1_end = span1_start + len(span1) - 1
                        context += span1
                        token_idx += len(span1)
                        i += l
                    # span2
                    elif t_id == target_t_ids[0]:
                        for l in range(len(target_t_ids)):
                            span2 += tokens[i + l][1] + " "
                        # setting span2 start and end indexes
                        span2_start = copy.deepcopy(token_idx)
                        span2_end = span2_start + len(span2) - 1
                        context += span2
                        token_idx += len(span2)
                        i += l
                    else:
                        context += token_text + " "
                        token_idx += len(token_text) + 1
                i += 1

            # storing causal and non-causal info
            try:
                idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                           "signal": []}

                new_row = {
                    "original_id": '{}-{}'.format(doc, global_id),
                    "span1": [span1.strip()],
                    "span2": [span2.strip()],
                    "signal": [],
                    "context": context.strip('\n'), "idx": idx_val, "label": label,
                    "direction": direction,
                    "source": self.namexid["eventstorylines"],
                    "ann_file": doc, "split": int(row['split'])}
                global_id += 1

                if self._check_span_indexes(new_row) and label in [0, 1]:
                    data = data.append(new_row, ignore_index=True)
                else:
                    mismatch += 1

            except Exception as e:
                print("[crest-log] EventStoryLine. Detail: {}".format(e))

        logging.info("[crest] eventstorylines is converted.")

        return data, mismatch

    def convert_caters(self):
        folders_path = self.dir_path + "caters/"

        global mismatch
        mismatch = 0

        def _get_context_spans(tags, doc_segments, arg1_id, arg2_id):

            df_columns = ['start', 'end', 'tag']
            df = pd.DataFrame(columns=df_columns)

            spans_tokens = {"span1": (' '.join(tags[arg1_id][0].split(' ')[1:]).strip()).split(';'),
                            "span2": (' '.join(tags[arg2_id][0].split(' ')[1:]).strip()).split(';')}

            for k, v in spans_tokens.items():
                for item in v:
                    arg = item.split(' ')
                    df = df.append(pd.DataFrame(
                        [[int(arg[0]), int(arg[1]), k]],
                        columns=df_columns
                    ), ignore_index=True)

            df = df.sort_values(by=['start'])

            i = 0
            while i < len(doc_segments):
                segment_idx = doc_segments[i][1]
                try:
                    if segment_idx <= df.iloc[0]["start"] and (i + 1 == len(doc_segments) or
                                                               (i + 1 < len(doc_segments) and df.iloc[0]["start"] <
                                                                doc_segments[i + 1][1])):
                        context = doc_segments[i][0]
                        newline_start = False
                        if context.startswith("\n"):
                            newline_start = True

                        spans = {"span1": [[], []], "span2": [[], []]}

                        for k, v in spans.items():
                            df_span = df.loc[df["tag"] == k]
                            for index, value in df_span.iterrows():
                                start = value["start"] - segment_idx
                                end = value["end"] - segment_idx
                                if newline_start:
                                    spans[k][1].append([start - 1, end - 1])
                                else:
                                    spans[k][1].append([start, end])
                                spans[k][0].append(context[start:end])

                        span1_text = tags[arg1_id][1]
                        span2_text = tags[arg2_id][1]

                        assert (" ".join(spans["span1"][0])).strip() == span1_text, (" ".join(
                            spans["span2"][0])).strip() == span2_text
                    i += 1
                except Exception as e:
                    print("[crest-log] CaTeRS. Detail: {}".format(e))

            return [spans["span1"][0], spans["span1"][1]], [spans["span2"][0], spans["span2"][1]], context.strip("\n")

        def extract_samples(folders, split):
            samples = pd.DataFrame(columns=self.scheme_columns)
            for folder in folders:
                docs_path = folders_path + "/" + folder
                docs = os.listdir(docs_path)
                for doc in docs:
                    tags = {}
                    if ".ann" in doc:
                        with open(docs_path + "/" + doc, 'r') as f:
                            lines = f.readlines()

                            # reading all tags information
                            for line in lines:
                                line_cols = line.split('\t')
                                tags[line_cols[0]] = [line_cols[1], line_cols[2].replace('\n', '')]

                            # reading the corresponding text file for the .ann file
                            with open(docs_path + "/" + doc.strip(".ann") + ".txt", 'r') as f:
                                doc_lines = f.read()

                            separator = "***"
                            doc_segments = doc_lines.split(separator)
                            context_segments = []
                            context_idx = 0
                            for segment in doc_segments:
                                context_segments.append([segment, context_idx])
                                context_idx += len(segment) + len(separator)

                            causal_tags = ["CAUSE_BEFORE", "CAUSE_OVERLAPS", "ENABLE_BEFORE", "ENABLE_OVERLAPS",
                                           "PREVENT_BEFORE", "PREVENT_OVERLAPS",
                                           "CAUSE_TO_END_BEFORE", "CAUSE_TO_END_OVERLAPS", "CAUSE_TO_END_DURING"]

                            # iterate through causal tags
                            for key, value in tags.items():
                                try:
                                    if key.startswith("R"):
                                        args = value[0].split(' ')
                                        original_id = key
                                        arg1_id = args[1].split(':')[1]
                                        arg2_id = args[2].split(':')[1]

                                        span1, span2, context = _get_context_spans(tags, context_segments, arg1_id,
                                                                                   arg2_id)

                                        idx_val = {"span1": span1[1],
                                                   "span2": span2[1],
                                                   "signal": []}

                                        direction = 0
                                        if any(causal_tag in value[0] for causal_tag in causal_tags):
                                            label = 1
                                        else:
                                            label = 0

                                        new_row = {"original_id": original_id, "span1": span1[0], "span2": span2[0],
                                                   "signal": [],
                                                   "context": context.strip('\n'),
                                                   "idx": idx_val, "label": label, "direction": direction,
                                                   "source": self.namexid["caters"],
                                                   "ann_file": doc,
                                                   "split": split}

                                        if self._check_span_indexes(new_row):
                                            samples = samples.append(new_row, ignore_index=True)
                                        else:
                                            mismatch += 1

                                except Exception as e:
                                    print("[crest-log] Error in converting CaTeRS. Detail: {}".format(e))
            return samples

        data = pd.DataFrame(columns=self.scheme_columns)
        data = data.append(extract_samples(["caters_test/test"], 2))
        data = data.append(extract_samples(["caters_evaluation/dev"], 1))
        data = data.append(extract_samples(["caters_evaluation/train"], 0))

        logging.info("[crest] caters is converted.")

        return data, mismatch

    def convert_because(self):
        """
        reading BECAUSE v2.1 Data
        :return:
        """

        nlp = spacy.load("en_core_web_sm")

        mismatch = 0

        # for NYT and PTB LDC subscription is needed to get access to the raw text.
        folders = ["CongressionalHearings", "MASC"]
        folders_path = self.dir_path + "BECAUSE-2.1/"

        data = pd.DataFrame(columns=self.scheme_columns)

        for folder in folders:
            docs_path = folders_path + folder
            docs = os.listdir(docs_path)
            for doc in docs:
                tags = {}
                if ".ann" in doc:
                    with open(docs_path + "/" + doc, 'r') as f:
                        lines = f.readlines()

                        # reading all tags information from .ann file
                        for line in lines:
                            line_cols = line.split('\t')
                            tags[line_cols[0]] = [line_cols[1], ""]
                            if len(line_cols) == 3:
                                tags[line_cols[0]][1] = line_cols[2].strip()

                    # reading the corresponding text file for the .ann file
                    with open(docs_path + "/" + doc.strip(".ann") + ".txt", 'r') as f:
                        doc_string = f.read()

                    # now, reading causal relations info
                    for key, value in tags.items():
                        try:
                            original_id = key

                            # causal samples
                            causal_tags = ["Consequence", "Purpose", "Motivation", "NonCausal"]
                            arg_tags = ["Effect", "Cause", "Arg0", "Arg1"]

                            if key.startswith("E") and any(
                                    causal_tag in value[0] for causal_tag in causal_tags) and any(
                                arg_tag in value[0] for arg_tag in arg_tags):
                                args = value[0].split(' ')
                                signal_id = args[0].split(':')[1].replace('\n', '')

                                direction = 0

                                # check if both arguments are available
                                if len(args) > 2:
                                    arg0_id = args[1].split(':')[1].replace('\n', '')
                                    arg1_id = args[2].split(':')[1].replace('\n', '')

                                    if 'Arg0:' in args[1] or 'Cause' in args[1]:
                                        direction = 0
                                    elif 'Arg1:' in args[1] or 'Effect' in args[1]:
                                        direction = 1
                                else:
                                    arg0_id = args[1].split(':')[1].replace('\n', '')
                                    arg1_id = ""

                                # ============================================
                                # extracting text spans and context

                                df_columns = ['start', 'end', 'tag']
                                df = pd.DataFrame(columns=df_columns)

                                arg0 = (' '.join(tags[arg0_id][0].split(' ')[1:]).strip()).split(';')

                                if arg1_id != "":
                                    arg1 = (' '.join(tags[arg1_id][0].split(' ')[1:]).strip()).split(';')
                                    for arg in arg1:
                                        df = df.append(pd.DataFrame(
                                            [[int(arg.split(' ')[0]), int(arg.split(' ')[1]), "span2"]],
                                            columns=df_columns
                                        ), ignore_index=True)

                                signal = (' '.join(tags[signal_id][0].split(' ')[1:]).strip()).split(';')

                                for arg in arg0:
                                    df = df.append(pd.DataFrame(
                                        [[int(arg.split(' ')[0]), int(arg.split(' ')[1]), "span1"]],
                                        columns=df_columns
                                    ), ignore_index=True)

                                for arg in signal:
                                    df = df.append(pd.DataFrame(
                                        [[int(arg.split(' ')[0]), int(arg.split(' ')[1]), "signal"]],
                                        columns=df_columns
                                    ), ignore_index=True)

                                df = df.sort_values('start')

                                doc_segments = []
                                for sen in list(nlp(doc_string).sents):
                                    doc_segments.append([sen.text_with_ws, sen.start_char])

                                context = ""
                                token_idx = 0

                                global_start = df.iloc[0]["start"]
                                global_end = df.iloc[len(df) - 1]["end"]

                                # building context
                                i = 0
                                while i < len(doc_segments):
                                    segment_idx = doc_segments[i][1]
                                    if segment_idx <= global_start and (i + 1 == len(doc_segments) or
                                                                        (i + 1 < len(doc_segments) and global_start <
                                                                         doc_segments[i + 1][1])):
                                        token_idx = copy.deepcopy(segment_idx)
                                        while (i + 1 == len(doc_segments)) or (
                                                i + 1 < len(doc_segments) and global_end > doc_segments[i][1]):
                                            context += doc_segments[i][0]
                                            i += 1
                                        break
                                    i += 1

                                spans = {"span1": [], "span2": [], "signal": [], "span1_idxs": [], "span2_idxs": [],
                                         "signal_idxs": []}

                                for index, row in df.iterrows():
                                    spans[row["tag"]].append(doc_string[row["start"]:row["end"]])
                                    start = row["start"] - token_idx
                                    end = row["end"] - token_idx
                                    spans[row["tag"] + "_idxs"].append([start, end])

                                span1 = [spans["span1"], spans["span1_idxs"]]
                                span2 = [spans["span2"], spans["span2_idxs"]]
                                signal = [spans["signal"], spans["signal_idxs"]]
                                # ============================================

                                # specifying the label
                                if "NonCausal" in value[0]:
                                    label = 0
                                else:
                                    label = 1

                                idx_val = {"span1": span1[1],
                                           "span2": span2[1],
                                           "signal": signal[1]}

                                row = {"original_id": original_id, "span1": span1[0], "span2": span2[0],
                                       "signal": signal[0],
                                       "context": context,
                                       "idx": idx_val, "label": label, "direction": direction,
                                       "source": self.namexid["because"],
                                       "ann_file": doc,
                                       "split": ""}

                                if self._check_span_indexes(row):
                                    data = data.append(row, ignore_index=True)
                                else:
                                    mismatch += 1

                        except Exception as e:
                            print("[crest-log] {}".format(e))

        logging.info("[crest] because is converted.")

        return data, mismatch

    def convert_copa(self):
        """
        converting Choice of Plausible Alternatives (COPA)
        :return:
        """
        folder_path = self.dir_path + "COPA-resources/datasets/"
        files = ["dev", "test"]
        mismatch = 0
        data = pd.DataFrame(columns=self.scheme_columns)

        for file in files:
            try:
                parser = ET.XMLParser(encoding="utf-8")
                tree = ET.parse(folder_path + "copa-" + file + ".xml", parser=parser)
                root = tree.getroot()

                for item in root.findall("./item"):
                    spans = {0: item[0].text, 1: item[1].text, 2: item[2].text}
                    original_id = item.attrib['id']

                    if int(item.attrib["most-plausible-alternative"]) == 1:
                        span_neg = spans[2]
                    else:
                        span_neg = spans[1]

                    span1 = spans[0]
                    span2 = spans[int(item.attrib["most-plausible-alternative"])]

                    if item.attrib["asks-for"] == "cause":
                        direction = 1
                    elif item.attrib["asks-for"] == "effect":
                        direction = 0

                    # final samples
                    pairs = [[span1, span2, 1, direction], [span1, span_neg, 0, direction]]

                    if file == "dev":
                        split = 1
                    elif file == "test":
                        split = 2

                    for pair in pairs:
                        context = pair[0] + " " + pair[1]
                        span1_start = 0
                        span1_end = len(pair[0]) - 1
                        span2_start = span1_end + 2
                        span2_end = span2_start + len(pair[1]) - 1

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                   "signal": []}

                        new_row = {"original_id": int(original_id), "span1": [pair[0].strip('.')],
                                   "span2": [pair[1].strip('.')],
                                   "signal": [],
                                   "context": context.strip('\n'),
                                   "idx": idx_val, "label": pair[2], "direction": direction,
                                   "source": self.namexid["copa"],
                                   "ann_file": "copa-" + file + ".xml",
                                   "split": split}

                        if self._check_span_indexes(new_row):
                            data = data.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1

            except Exception as e:
                print("[crest-log] COPA. Detail: {}".format(e))

        logging.info("[crest] copa is converted.")

        return data, mismatch

    def convert_pdtb3(self):
        mismatch = 0

        # reading pdtb3 into dataframe
        df = pd.read_excel(self.dir_path + 'pdtb3.xlsx')

        nlp = spacy.load("en_core_web_sm")

        classes = ['Contingency.Cause.Result', 'Contingency.Cause.Reason']

        data = pd.DataFrame(columns=self.scheme_columns)

        for idx, row in df.iterrows():
            if isinstance(row['SClass1A'], str):
                try:
                    if row['SClass1A'] in classes and ';' not in row['Arg1SpanList'] and ';' not in row['Arg2SpanList']:
                        arg1 = row['Arg1SpanList'].split('..')
                        arg2 = row['Arg2SpanList'].split('..')
                        arg1_s, arg1_e = int(arg1[0]), int(arg1[1])
                        arg2_s, arg2_e = int(arg2[0]), int(arg2[1])
                        full_text = row['FullRawText']

                        arg1_text = full_text[arg1_s:arg1_e]
                        arg2_text = full_text[arg2_s:arg2_e]

                        # ============================================
                        # specifying the direction
                        if row['SClass1A'] == 'Contingency.Cause.Result':
                            if arg1_e < arg2_s:
                                direction = 0
                            else:
                                direction = 1
                        else:
                            if arg1_e < arg2_s:
                                direction = 1
                            else:
                                direction = 0

                        # ============================================
                        # extracting text spans and context

                        df_columns = ['start', 'end', 'text']
                        df_args = pd.DataFrame(columns=df_columns)

                        df_args = df_args.append(pd.DataFrame([[arg1_s, arg1_e, arg1_text]], columns=df_columns),
                                                 ignore_index=True)
                        df_args = df_args.append(pd.DataFrame([[arg2_s, arg2_e, arg2_text]], columns=df_columns),
                                                 ignore_index=True)

                        df_args = df_args.sort_values('start')

                        doc_segments = []
                        for sen in list(nlp(full_text).sents):
                            doc_segments.append([sen.text_with_ws, sen.start_char])

                        context = ""
                        token_idx = 0

                        global_start = df_args.iloc[0]["start"]
                        global_end = df_args.iloc[1]["end"]

                        # building context
                        i = 0
                        while i < len(doc_segments):
                            segment_idx = doc_segments[i][1]
                            if segment_idx <= global_start and (i + 1 == len(doc_segments) or (
                                    i + 1 < len(doc_segments) and global_start < doc_segments[i + 1][1])):
                                token_idx = copy.deepcopy(segment_idx)
                                while (i + 1 == len(doc_segments)) or (
                                        i + 1 < len(doc_segments) and global_end > doc_segments[i][1]):
                                    context += doc_segments[i][0]
                                    i += 1
                                break
                            i += 1

                        # ===========================================
                        # saving relations information
                        idx_val = {
                            "span1": [[df_args.iloc[0]['start'] - token_idx, df_args.iloc[0]['end'] - token_idx]],
                            "span2": [[df_args.iloc[1]['start'] - token_idx, df_args.iloc[1]['end'] - token_idx]],
                            "signal": []}

                        new_row = {"original_id": int(row['RelationId']),
                                   "span1": [df_args.iloc[0]['text']],
                                   "span2": [df_args.iloc[1]['text']],
                                   "signal": [],
                                   "context": context.strip('\n'),
                                   "idx": idx_val, "label": 1, "direction": direction,
                                   "source": self.namexid["pdtb3"],
                                   "ann_file": "",
                                   "split": 0}

                        if self._check_span_indexes(new_row):
                            data = data.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1
                except Exception as e:
                    print("[crest-log] PDTB3. Detail: {}".format(e))

        logging.info("[crest] PDTB3 is converted.")

        return data, mismatch

    @staticmethod
    def _get_between_text(str_1, str_2, orig_text):
        result = re.search(str_1 + "(.*)" + str_2, orig_text)
        return result.group(1)

    @staticmethod
    def _check_span_indexes(row):
        """
        checking if spans/signal indexes are correctly stored
        :param row:
        :return:
        """

        span1 = ""
        span2 = ""
        signal = ""
        try:
            for arg in row["idx"]["span1"]:
                span1 += row["context"][arg[0]:arg[1]] + " "

            for arg in row["idx"]["span2"]:
                span2 += row["context"][arg[0]:arg[1]] + " "

            for sig in row["idx"]["signal"]:
                signal += row["context"][sig[0]:sig[1]] + " "

            FLAGS = {'s1': False, 's2': False, 'sig': False, 'context': False}
            if span1.strip() != (" ".join(row["span1"])).strip():
                print("span1: [{}]\n[{}]".format(span1, (" ".join(row["span1"])).strip()))
                FLAGS["s1"] = True
            if span2.strip() != (" ".join(row["span2"])).strip():
                print("span2: [{}]\n[{}]".format(span2, (" ".join(row["span2"])).strip()))
                FLAGS["s2"] = True
            if signal.strip() != (" ".join(row["signal"])).strip():
                print("signal: [{}]\n[{}]".format(signal, (" ".join(row["signal"])).strip()))
                FLAGS["sig"] = True
            if str(row["context"]) == "nan":
                FLAGS["context"] = True
            if any(a for a in FLAGS.values()):
                print("context: [{}] \n========".format(row["context"]))
                return False
        except Exception:
            return False
        return True
