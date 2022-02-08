import re
import os
import ast
import sys
import copy
import json
import spacy
import logging
import pandas as pd
from xml.dom import minidom
import xml.etree.ElementTree as ET

from utils import idx_to_string


class Converter:
    """
    idx = {'span1': [], 'span2': [], 'signal': []} -> indexes of span1, span2, and signal tokens/spans in context
    each value in the idx dictionary is a list of lists of indexes. For example, if span1 has multiple tokens in context
    with start:end indexes 2:5 and 10:13, respectively, span1's value in 'idx' will be [[2, 5],[10, 13]]. Lists are
    sorted based on the start indexes of tokens. Same applies for span2 and signal.
    -------------------------------------------------------------------------------
    label => 0: non-causal, 1: causal
    direction => 0: span1 => span2, 1: span2 => span1, -1: not-specified
    NOTE: span1 always precedes span2 in context
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
                        "biocause": 10,
                        "tcr": 11,
                        "ade": 12,
                        "semeval_2020_5": 13
                        }

        self.idxmethod = {self.namexid["semeval_2007_4"]: self.convert_semeval_2007_4,
                          self.namexid["semeval_2010_8"]: self.convert_semeval_2010_8,
                          self.namexid["event_causality"]: self.convert_event_causality,
                          self.namexid["causal_timebank"]: self.convert_causal_timebank,
                          self.namexid["eventstorylines"]: self.convert_eventstorylines_v1,
                          self.namexid["caters"]: self.convert_caters,
                          self.namexid["because"]: self.convert_because,
                          self.namexid["copa"]: self.convert_copa,
                          self.namexid["pdtb3"]: self.convert_pdtb3,
                          self.namexid["biocause"]: self.convert_biocause,
                          self.namexid["tcr"]: self.convert_tcr,
                          self.namexid["ade"]: self.convert_ade,
                          self.namexid["semeval_2020_5"]: self.convert_semeval_2020_5
                          }

        # causal tags in PDTB3
        self.pdtb3_causal_classes = ['Contingency.Cause.Result', 'Contingency.Cause.Reason',
                                     'Contingency.Cause+Belief.Reason+Belief', 'Contingency.Cause+Belief.Result+Belief',
                                     'Contingency.Cause+SpeechAct.Reason+SpeechAct',
                                     'Contingency.Cause+SpeechAct.Result+SpeechAct']

        # temporal tags in PDTB3
        self.pdtb3_temporal_classes = ['Temporal.Asynchronous.Precedence', 'Temporal.Asynchronous.Succession',
                                       'Temporal.Synchronous']

    def convert2crest(self, dataset_ids=[], save_file=False):
        """
        Converting a dataset to CREST
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

        # updating global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data['global_id'] = global_ids
        data.reset_index()

        if save_file:
            data.to_excel(self.dir_path + "crest.xlsx")

        return data, total_mis

    def convert_semeval_2007_4(self):
        """
        Converting SemEval 2007 task 4 data
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
                        if self.check_span_indexes(new_row) and span1_end < span2_start:
                            new_row["idx"] = idx_to_string(new_row["idx"])
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
                    self.dir_path + 'semeval_2007_4/task-4-training/relation-{}-train.txt'.format(str(relation_id)),
                    mode='r',
                    encoding='cp1252') as train:
                train_content = train.readlines()

            # this is the test set
            with open(self.dir_path + 'semeval_2007_4/task-4-scoring/relation-{}-score.txt'.format(str(relation_id)),
                      mode='r',
                      encoding='cp1252') as key:
                test_content = key.readlines()

            data = data.append(extract_samples(train_content, 0))
            data = data.append(extract_samples(test_content, 2))

        logging.info("[crest] semeval_2007_4 is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

        return data, mismatch

    def convert_semeval_2010_8(self):
        """
        Converting SemEval 2010 task 8 data
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

                        if self.check_span_indexes(new_row) and span1_end < span2_start:
                            new_row["idx"] = idx_to_string(new_row["idx"])
                            samples = samples.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1

                    except Exception as e:
                        print("[crest-log] Incorrect formatting for semeval10-task8 record. Detail: " + str(e))
            return samples

        # reading files
        with open(self.dir_path + 'semeval_2010_8/SemEval2010_task8_training/TRAIN_FILE.txt', mode='r',
                  encoding='cp1252') as train:
            train_content = train.readlines()

        with open(self.dir_path + 'semeval_2010_8/SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt',
                  mode='r', encoding='cp1252') as key:
            test_content = key.readlines()

        data = pd.DataFrame(columns=self.scheme_columns)

        data = data.append(extract_samples(train_content, 0))
        data = data.append(extract_samples(test_content, 2))

        logging.info("[crest] semeval_2010_8 is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

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
                        # reading causal tags
                        if tag.startswith("C") or tag.startswith("R"):
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
        data_path = self.dir_path + "event_causality/"

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

                        if self.check_span_indexes(new_row):
                            new_row["idx"] = idx_to_string(new_row["idx"])
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

                        if self.check_span_indexes(new_row):
                            new_row["idx"] = idx_to_string(new_row["idx"])
                            data = data.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1

        logging.info("[crest] event_causality is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

        return data, mismatch

    def convert_causal_timebank(self):
        """
        Converting samples from Causal-TimeBank
        """
        mismatch = 0
        data_path = self.dir_path + "causal_timebank/Causal-TimeBank-CAT"
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
                           "context": context.strip('\n'), "idx": idx_val, "label": 1,
                           "direction": direction,
                           "source": self.namexid["causal_timebank"],
                           "ann_file": file, "split": ""}

                if self.check_span_indexes(new_row):
                    new_row["idx"] = idx_to_string(new_row["idx"])
                    data = data.append(new_row, ignore_index=True)
                else:
                    mismatch += 1

        logging.info("[crest] causal_timebank is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

        return data, mismatch

    def convert_eventstorylines_v1(self, version="1.5"):
        """
        Converting causal and non-causal samples from EventStoryLines
        """
        mismatch = 0
        docs_path = self.dir_path + "eventstorylines/annotated_data/v" + version

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

                                # making sure span1 precedes span2 in context
                                if span2_start < span1_start:
                                    span1_start, span2_start = span2_start, span1_start
                                    span1_end, span2_end = span2_end, span1_end
                                    span1, span2 = span2, span1
                                    direction = 0 if direction == 1 else 1

                                # storing causal and non-causal info
                                try:
                                    idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                               "signal": []}

                                    new_row = {"original_id": original_id, "span1": [span1.strip()],
                                               "span2": [span2.strip()],
                                               "signal": [],
                                               "context": context.strip('\n'), "idx": idx_val,
                                               "label": label,
                                               "direction": direction,
                                               "source": self.namexid["eventstorylines"],
                                               "ann_file": doc, "split": ""}

                                    if self.check_span_indexes(new_row):
                                        new_row["idx"] = idx_to_string(new_row["idx"])
                                        data = data.append(new_row, ignore_index=True)
                                    else:
                                        mismatch += 1

                                except Exception as e:
                                    print("[crest-log] EventStoryLine. Detail: {}".format(e))

        logging.info("[crest] eventstorylines is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

        return data, mismatch

    def convert_eventstorylines_v2(self, version="1.5"):
        """
        Converting causal and non-causal samples from EventStoryLines based on evaluation_format file
        """

        splits = {'full_corpus/v{}/event_mentions_extended'.format(version): 0,
                  'test_corpus/v{}/event_mentions_extended'.format(version): 2}

        annotations = pd.DataFrame(columns=['file', 'source', 'target', 'label', 'split'])

        # creating a dictionary of all documents
        data = pd.DataFrame(columns=self.scheme_columns)

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
                    "original_id": '{}'.format(doc),
                    "span1": [span1.strip()],
                    "span2": [span2.strip()],
                    "signal": [],
                    "context": context.strip('\n'), "idx": idx_val, "label": label,
                    "direction": direction,
                    "source": self.namexid["eventstorylines"],
                    "ann_file": doc, "split": int(row['split'])}

                if self.check_span_indexes(new_row) and label in [0, 1]:
                    new_row["idx"] = idx_to_string(new_row["idx"])
                    data = data.append(new_row, ignore_index=True)
                else:
                    mismatch += 1

            except Exception as e:
                print("[crest-log] EventStoryLine. Detail: {}".format(e))

        logging.info("[crest] eventstorylines is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

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
                            with open(docs_path + "/" + doc[:-4] + ".txt", 'r') as f:
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
                                                   "idx": idx_val, "label": label,
                                                   "direction": direction,
                                                   "source": self.namexid["caters"],
                                                   "ann_file": doc,
                                                   "split": split}

                                        if self.check_span_indexes(new_row):
                                            new_row["idx"] = idx_to_string(new_row["idx"])
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

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

        return data, mismatch

    def convert_because(self):
        """
        Converting BECAUSE v2.1 Data
        :return:
        """

        nlp = spacy.load("en_core_web_sm")

        mismatch = 0

        # for NYT and PTB LDC subscription is needed to get access to the raw text.
        folders = ["CongressionalHearings", "MASC"]
        folders_path = self.dir_path + "because/"

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
                    with open(docs_path + "/" + doc[:-4] + ".txt", 'r') as f:
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

                                if self.check_span_indexes(row):
                                    row["idx"] = idx_to_string(row["idx"])
                                    data = data.append(row, ignore_index=True)
                                else:
                                    mismatch += 1

                        except Exception as e:
                            print("[crest-log] {}".format(e))

        logging.info("[crest] because is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

        return data, mismatch

    def convert_copa(self, dataset_code=1):
        """
        Converting Choice of Plausible Alternatives (COPA) and its variations
        :param dataset_code: an integer to specify the variation of COPA. Supported values:
        dataset_code: 1 -> original COPA dataset by Melissa Roemmele, et al. - 2011
        dataset_code: 2 -> BCOPA-CE dataset by Mingyue Han, et al. 2021
        :return:
        """

        split_code = {"dev": 1, "test": 2}

        folder_path = self.dir_path + "copa/datasets/"

        if dataset_code == 1:
            files = {"dev": "copa-dev.xml", "test": "copa-test.xml"}
        elif dataset_code == 2:
            files = {"test": "BCOPA-CE.xml"}

        mismatch = 0

        data = pd.DataFrame(columns=self.scheme_columns)

        for split_name, file_path in files.items():
            try:
                parser = ET.XMLParser(encoding="utf-8")
                tree = ET.parse(folder_path + file_path, parser=parser)
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

                    for pair in pairs:
                        context = pair[0].strip() + " " + pair[1].strip()
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
                                   "ann_file": file_path,
                                   "split": split_code[split_name]}

                        if self.check_span_indexes(new_row):
                            new_row["idx"] = idx_to_string(new_row["idx"])
                            data = data.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1
                            print(new_row)

            except Exception as e:
                print("[crest-log] COPA. Detail: {}".format(e))

        logging.info("[crest] copa is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

        return data, mismatch

    def convert_pdtb3(self, tag_class='causal'):
        mismatch = 0

        # reading pdtb3 into dataframe
        df = pd.read_excel(self.dir_path + 'pdtb3.xlsx')

        nlp = spacy.load("en_core_web_sm")

        data = pd.DataFrame(columns=self.scheme_columns)

        if tag_class == 'causal':
            relation_tags = self.pdtb3_causal_classes
        elif tag_class == 'temporal':
            relation_tags = self.pdtb3_temporal_classes

        for idx, row in df.iterrows():
            if isinstance(row['SClass1A'], str):
                try:
                    if row['SClass1A'] in relation_tags and ';' not in row['Arg1SpanList'] and ';' not in row[
                        'Arg2SpanList']:
                        arg1 = row['Arg1SpanList'].split('..')
                        arg2 = row['Arg2SpanList'].split('..')
                        arg1_s, arg1_e = int(arg1[0]), int(arg1[1])
                        arg2_s, arg2_e = int(arg2[0]), int(arg2[1])
                        full_text = row['FullRawText']

                        arg1_text = full_text[arg1_s:arg1_e]
                        arg2_text = full_text[arg2_s:arg2_e]

                        # ============================================
                        # specifying the direction
                        if 'Result' in row['SClass1A']:
                            direction = 0 if arg1_e < arg2_s else 1
                        else:
                            direction = 1 if arg1_e < arg2_s else 0

                        # ============================================
                        # extracting text spans and context

                        df_columns = ['start', 'end', 'text']
                        df_args = pd.DataFrame(columns=df_columns)

                        df_args = df_args.append(pd.DataFrame([[arg1_s, arg1_e, arg1_text]], columns=df_columns),
                                                 ignore_index=True)
                        df_args = df_args.append(pd.DataFrame([[arg2_s, arg2_e, arg2_text]], columns=df_columns),
                                                 ignore_index=True)

                        df_args = df_args.sort_values('start')

                        # updating direction if needed, since we sort the start indexes
                        if arg1_s != df_args.iloc[0].start:
                            direction = 0 if direction == 1 else 1

                        sents = []
                        for sen in list(nlp(full_text).sents):
                            sents.append([sen.text_with_ws, sen.start_char])

                        assert ''.join([t[0] for t in sents]) == full_text

                        context = ""
                        token_idx = 0
                        start_sent_idx = df_args.iloc[0]["start"]
                        end_sent_idx = df_args.iloc[1]["end"]

                        # adding more context by considering 1 extra sentence before and after
                        if start_sent_idx >= 1:
                            start_sent_idx -= 1
                        if end_sent_idx < len(sents) - 1:
                            end_sent_idx += 1

                        # cutting unnecessary sentences from context which also
                        i = 0
                        while i < len(sents):
                            start_idx = sents[i][1]
                            if start_idx <= start_sent_idx and (i == len(sents) - 1 or (
                                    i < len(sents) - 1 and start_sent_idx < sents[i + 1][1])):
                                token_idx = copy.deepcopy(start_idx)
                                while (i == len(sents) - 1) or (
                                        i < len(sents) - 1 and sents[i][1] < end_sent_idx):
                                    context += sents[i][0]
                                    i += 1
                                break
                            i += 1

                        assert context in full_text

                        # removing leading spaces
                        len_context_pre = len(context)
                        context = context.lstrip()
                        len_context_post = len(context)
                        diff = len_context_pre - len_context_post

                        # ===========================================
                        # saving relations information based on new context indexes
                        idx_val = {
                            "span1": [[df_args.iloc[0]['start'] - token_idx - diff,
                                       df_args.iloc[0]['end'] - token_idx - diff]],
                            "span2": [[df_args.iloc[1]['start'] - token_idx - diff,
                                       df_args.iloc[1]['end'] - token_idx - diff]],
                            "signal": []}

                        s1_text, s1_s, s1_e = self._normalize_span(df_args.iloc[0]['text'], idx_val['span1'][0][0],
                                                                   idx_val['span1'][0][1])
                        s2_text, s2_s, s2_e = self._normalize_span(df_args.iloc[1]['text'], idx_val['span2'][0][0],
                                                                   idx_val['span2'][0][1])
                        idx_val['span1'] = [[s1_s, s1_e]]
                        idx_val['span2'] = [[s2_s, s2_e]]

                        new_row = {"original_id": int(row['RelationId']),
                                   "span1": [s1_text],
                                   "span2": [s2_text],
                                   "signal": [],
                                   "context": context.strip(),
                                   "idx": idx_val, "label": 1, "direction": direction,
                                   "source": self.namexid["pdtb3"],
                                   "ann_file": "",
                                   "split": 0}

                        if self.check_span_indexes(new_row):
                            new_row["idx"] = idx_to_string(new_row["idx"])
                            data = data.append(new_row, ignore_index=True)
                        else:
                            mismatch += 1
                except Exception as e:
                    print("[crest-log] PDTB3. Detail: {}".format(e))

        logging.info("[crest] PDTB3 is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

        return data, mismatch

    def convert_biocause(self):
        """
        Converting causal relations from BioCause corpus
        :return:
        """
        folders_path = self.dir_path + "biocause/"

        global mismatch
        mismatch = 0

        def _get_context_spans(tags, doc_context, rel_direction, arg1_id, arg2_id):
            df_columns = ['start', 'end', 'tag', 'text']

            df = pd.DataFrame(columns=df_columns)

            spans_tokens = {"span1": tags[arg1_id][0].split(' ')[1:],
                            "span2": tags[arg2_id][0].split(' ')[1:]}

            spans_text = {"span1": tags[arg1_id][1], "span2": tags[arg2_id][1]}

            for k, v in spans_tokens.items():
                df = df.append(pd.DataFrame(
                    [[int(v[0]), int(v[1]), k, spans_text[k]]],
                    columns=df_columns
                ), ignore_index=True)

            span_idx = df.iloc[0]['start']

            df = df.sort_values(by=['start'])

            # checking if the spans order is changed. If yes, change the direction accordingly
            if span_idx != df.iloc[0]['start']:
                rel_direction = 1 if rel_direction == 0 else 0
                spans_text['span1'], spans_text['span2'] = spans_text['span2'], spans_text['span1']

            span1_text = doc_context[df.iloc[0]['start']:df.iloc[0]['end']]
            span2_text = doc_context[df.iloc[1]['start']:df.iloc[1]['end']]

            assert span1_text == spans_text['span1']
            assert span2_text == spans_text['span2']

            return [[span1_text], [[df.iloc[0]['start'], df.iloc[0]['end']]]], [[span2_text], [
                [df.iloc[1]['start'], df.iloc[1]['end']]]], rel_direction

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
                                if line_cols[0].startswith('E'):
                                    tags[line_cols[0]] = [tag.replace('\n', '') for tag in line_cols[1].split(' ')]
                                elif line_cols[0].startswith('T'):
                                    tags[line_cols[0]] = [tag.replace('\n', '') for tag in line_cols[1:]]

                            # reading the corresponding text file for the .ann file
                            with open(docs_path + "/" + doc[:-4] + ".txt", 'r') as f:
                                context = f.read()

                            # iterate through causal tags
                            for key, value in tags.items():
                                try:
                                    if value[0].startswith("Causality:") and len(value) == 3:
                                        arg_signal = tags[value[0].split(':')[1]]
                                        arg1 = value[1].split(':')
                                        arg2 = value[2].split(':')

                                        # initialize direction value
                                        direction = 0 if arg1[0] in ['Cause', 'Evidence'] else 1

                                        span1, span2, direction = _get_context_spans(tags, context, direction, arg1[1],
                                                                                     arg2[1])

                                        idx_val = {"span1": span1[1],
                                                   "span2": span2[1],
                                                   "signal": [
                                                       [int(arg_signal[0].split(' ')[1]),
                                                        int(arg_signal[0].split(' ')[2])]]}

                                        signal_text = arg_signal[1] if len(arg_signal) == 2 else context[
                                                                                                 idx_val['signal'][0][
                                                                                                     0]:
                                                                                                 idx_val['signal'][0][
                                                                                                     1]]
                                        new_row = {"original_id": '', "span1": span1[0], "span2": span2[0],
                                                   "signal": [signal_text],
                                                   "context": context.strip('\n'),
                                                   "idx": idx_val, "label": 1, "direction": direction,
                                                   "source": self.namexid["biocause"],
                                                   "ann_file": doc,
                                                   "split": split}

                                        if self.check_span_indexes(new_row):
                                            new_row["idx"] = idx_to_string(new_row["idx"])
                                            samples = samples.append(new_row, ignore_index=True)
                                        else:
                                            mismatch += 1

                                except Exception as e:
                                    print("[crest-log] Error in converting BioCause. Detail: {}".format(e))
            return samples

        data = pd.DataFrame(columns=self.scheme_columns)
        data = data.append(extract_samples(["BioCause_corpus"], 0))

        logging.info("[biocause] caters is converted.")

        # adding global id to the data frame
        global_ids = [i for i in range(1, len(data) + 1)]
        data.insert(0, 'global_id', global_ids)
        data.reset_index()

        return data, mismatch

    def convert_tcr(self):
        """
        Converting the Temporal and Causal Reasoning (TCR) dataset.
        we only convert the C-Links (causal relations) from this dataset
        :return:
        """

        def read_text(file_name, event_a_id, event_b_id):
            idx_val = {"span1": [], "span2": [], "signal": []}
            parsed_doc = minidom.parse(self.dir_path + "tcr/TemporalPart/{}".format(file_name))
            elements = parsed_doc.getElementsByTagName('TEXT')
            text = ""
            token_index = 0
            tagxid = {"EVENT": "eid", "TIMEX3": "tid"}
            for element in elements:
                if element.tagName == "TEXT":
                    for item in element.childNodes:
                        if item.nodeName == "#text":
                            text += item.wholeText
                            token_index += len(item.wholeText)
                        elif item.nodeName == "EVENT" or item.nodeName == "TIMEX3":
                            item_text = ' '.join([child_node.wholeText for child_node in item.childNodes])
                            text += item_text
                            start_end = [token_index, token_index + len(item_text)]
                            token_index += len(item_text)

                            if item.attributes[tagxid[item.nodeName]].value == event_a_id:
                                idx_val["span1"].append(start_end)
                                event_a_text = item_text
                            elif item.attributes[tagxid[item.nodeName]].value == event_b_id:
                                idx_val["span2"].append(start_end)
                                event_b_text = item_text
            return text, idx_val, [event_a_text, event_b_text]

        mismatch = 0
        data = pd.DataFrame(columns=self.scheme_columns)

        test_files = ["2010.01.08.facebook.bra.color", "2010.01.12.haiti.earthquake", "2010.01.12.turkey.israel",
                      "2010.01.13.google.china.exit", "2010.01.13.mexico.human.traffic.drug"]

        with open(self.dir_path + "tcr/CausalPart/allClinks.txt", 'r') as in_file:
            lines = in_file.readlines()

        annotations = [line.strip().split('\t') for line in lines]

        for annotation in annotations:
            file_path = annotation[0] + ".tml"
            text, idx_val, events_text = read_text(file_path, annotation[1], annotation[2])
            direction = 1 if annotation[3] == "caused_by" else 0

            split = 2 if annotation[0] in test_files else 1

            # saving the sample
            new_row = {"original_id": '', "span1": [events_text[0]], "span2": [events_text[1]], "signal": [],
                       "context": text,
                       "idx": idx_val, "label": 1, "direction": direction,
                       "source": self.namexid["tcr"],
                       "ann_file": file_path,
                       "split": split}

            if self.check_span_indexes(new_row):
                new_row["idx"] = idx_to_string(new_row["idx"])
                data = data.append(new_row, ignore_index=True)
            else:
                mismatch += 1
        return data, mismatch

    def convert_ade(self):
        """
        Converting Benchmark Corpus for Adverse Drug Effects (ADE) to CREST
        :return:
        """
        mismatch = 0
        data = pd.DataFrame(columns=self.scheme_columns)

        # Title and Abstract in the following file are retrieved from the PubMed in a separate process
        df = pd.read_excel(self.dir_path + 'ade/ade.xlsx')
        for idx, row in df.iterrows():
            header = "{}\n\n".format(row['PMID'])
            text = "{}{}\n\n{}".format(header, row['Title'], row['Abstract'])

            # since PMID and Title are part of text, we want to remove then to keep the relevant context
            e1_start = row['e1_start'] - len(header)
            e1_end = row['e1_end'] - len(header)
            e2_start = row['e2_start'] - len(header)
            e2_end = row['e2_end'] - len(header)

            text = text[len(header):]
            e1 = text[e1_start:e1_end]
            e2 = text[e2_start:e2_end]

            if e1 == row['e1_text'] and e2 == row['e2_text']:
                # since in all relations DRUG is the second span: [AE|DOSE] <- DRUG, we set direction to 1 by default
                # meaning that the relation is from DRUG (span2) to span1 (either AE or DOSE)
                direction = 1

                # however, we also want to make sure span1 appears before span2 in context
                if e1_start > e2_start:
                    e1_start, e2_start = e2_start, e1_start
                    e1_end, e2_end = e2_end, e1_end
                    e1, e2 = e2, e1
                    direction = 0

                idx_val = {"span1": [[e1_start, e1_end]], "span2": [[e2_start, e2_end]], "signal": []}

                # saving the sample
                new_row = {"original_id": '', "span1": [e1], "span2": [e2], "signal": [],
                           "context": text,
                           "idx": idx_val, "label": 1, "direction": direction,
                           "source": self.namexid["ade"],
                           "ann_file": "",
                           "split": 0}

                if self.check_span_indexes(new_row):
                    new_row["idx"] = idx_to_string(new_row["idx"])
                    data = data.append(new_row, ignore_index=True)
                else:
                    mismatch += 1
        return data, mismatch

    def convert_semeval_2020_5(self):
        """
        Converting the Subtask-2 data from SemEval 2020 Task 5:  Detecting Antecedent and Consequent (DAC)
        :return:
        """
        mismatch = 0
        data = pd.DataFrame(columns=self.scheme_columns)

        train_path = self.dir_path + "semeval_2020_5/Subtask-2/subtask2_train.csv"
        test_path = self.dir_path + "semeval_2020_5/Subtask-2/subtask2_test.csv"

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        splits = {0: train, 2: test}

        for split_code, split_data in splits.items():
            for idx, row in split_data.iterrows():
                direction = 0
                e1 = row['antecedent']
                e2 = row['consequent']
                context = row['sentence']
                e1_start = int(row['antecedent_startid'])
                e1_end = int(row['antecedent_endid']) + 1
                e2_start = int(row['consequent_startid'])
                e2_end = int(row['consequent_endid']) + 1

                if e1_start > e2_start:
                    e1_start, e2_start = e2_start, e1_start
                    e1_end, e2_end = e2_end, e1_end
                    e1, e2 = e2, e1
                    direction = 1

                idx_val = {"span1": [[e1_start, e1_end]], "span2": [[e2_start, e2_end]], "signal": []}
                e1 = '' if e1 == '{}' else e1
                e2 = '' if e2 == '{}' else e2

                # saving the sample
                new_row = {"original_id": '', "span1": [e1], "span2": [e2], "signal": [],
                           "context": context,
                           "idx": idx_val, "label": 1, "direction": direction,
                           "source": self.namexid["semeval_2020_5"],
                           "ann_file": "",
                           "split": split_code}

                # since some records may have a missing consequent, we check the indices not to be -1
                if self.check_span_indexes(new_row):
                    new_row["idx"] = idx_to_string(new_row["idx"])
                    data = data.append(new_row, ignore_index=True)
                else:
                    mismatch += 1
        return data, mismatch

    @staticmethod
    def _get_between_text(str_1, str_2, orig_text):
        result = re.search(str_1 + "(.*)" + str_2, orig_text)
        return result.group(1)

    @staticmethod
    def check_span_indexes(row, print_mismatch=False):
        """
        checking if spans/signal indexes are correctly stored
        :param row:
        :param print_mismatch:
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

            flags = {'s1': False, 's2': False, 'sig': False, 'context': False}
            if span1.strip() != (" ".join(row["span1"])).strip():
                if print_mismatch:
                    print("span1: [{}]\n[{}]".format(span1, (" ".join(row["span1"])).strip()))
                flags["s1"] = True
            if span2.strip() != (" ".join(row["span2"])).strip():
                if print_mismatch:
                    print("span2: [{}]\n[{}]".format(span2, (" ".join(row["span2"])).strip()))
                flags["s2"] = True
            if signal.strip() != (" ".join(row["signal"])).strip():
                if print_mismatch:
                    print("signal: [{}]\n[{}]".format(signal, (" ".join(row["signal"])).strip()))
                flags["sig"] = True
            if str(row["context"]) == "nan":
                flags["context"] = True
            if any(a for a in flags.values()):
                if print_mismatch:
                    print("context: [{}] \n========".format(row["context"]))
                return False
        except Exception as e:
            return False
        return True

    @staticmethod
    def _normalize_span(text, start, end):
        len_1 = len(text)
        text = text.lstrip()
        len_2 = len(text)
        start = start + abs(len_1 - len_2)

        len_1 = len(text)
        text = text.rstrip()
        len_2 = len(text)
        end = end - abs(len_1 - len_2)

        return text, start, end


def crest2tacred(df, output_file_name, split=[], source=[], no_order=False, save_json=False):
    """
    converting CREST-formatted data to TACRED (https://nlp.stanford.edu/projects/tacred/)
    :param df: pandas data frame of the CREST-formatted excel file
    :param output_file_name: name of output file without extension
    :param no_order: True if we want to remove spans order, False, otherwise
    :param save_json: binary value, True, if want to save result in a JSON file, False, otherwise
    :param split: split of the data, value is a list of numbers such as 0: train, 1: dev, test: 2. will return all data by default
    :param source: source of the data, a list of integer numbers
    :return: list of dictionaries
    """

    def get_token_indices(i_idx, t_idx, span_end_idx, all_tokens):
        span_tokens = []
        while t_idx < span_end_idx:
            span_tokens.append(all_tokens[i_idx])
            t_idx += len(all_tokens[i_idx])
            i_idx += 1
        return span_tokens, i_idx

    nlp = spacy.load("en_core_web_sm")

    if not type(df) == pd.core.frame.DataFrame:
        print("first parameter should be a pandas data frame")
        raise TypeError

    records = []
    excluded_rows = []
    excluded_records = []
    records_df = []
    for index, row in df.iterrows():
        try:
            idx = ast.literal_eval(str(row['idx']))
            # making sure spans are made of consecutive tokens
            if len(idx['span1']) == 1 and len(idx['span2']) == 1:
                record = {}
                span1_start = idx['span1'][0][0]
                span1_end = idx['span1'][0][1]

                span2_start = idx['span2'][0][0]
                span2_end = idx['span2'][0][1]

                if no_order:
                    if span2_start < span1_start:
                        span1_start, span2_start = span2_start, span1_start
                        span1_end, span2_end = span2_end, span1_end

                label = int(row['label'])
                direction = int(row['direction'])

                # creating list of tokens in context and finding spans' start and end indices
                token_idx = 0
                doc = nlp(row['context'])
                tokens = [token.text_with_ws for token in doc]

                for i in range(len(tokens)):
                    if token_idx == span1_start:
                        record['span1_start'] = copy.deepcopy(i)
                        span1_tokens, record['span1_end'] = get_token_indices(i, token_idx, span1_end, tokens)
                    elif token_idx == span2_start:
                        record['span2_start'] = copy.deepcopy(i)
                        span2_tokens, record['span2_end'] = get_token_indices(i, token_idx, span2_end, tokens)

                    token_idx += len(tokens[i])

                # getting the label and span type
                if direction == 0 or direction == -1:
                    record['direction'] = 'RIGHT'
                    record['span1_type'] = 'SPAN1'
                    record['span2_type'] = 'SPAN2'
                elif direction == 1:
                    record['direction'] = 'LEFT'
                    record['span1_type'] = 'SPAN2'
                    record['span2_type'] = 'SPAN1'

                record['id'] = str(row['global_id'])
                record['token'] = tokens
                record['relation'] = label
                features = ['id', 'token', 'span1_start', 'span1_end', 'span2_start', 'span2_end', 'relation']

                # check if record has all the required fields
                if all(feature in record for feature in features) and (
                        len(split) == 0 or int(row['split']) in split) and (
                        len(source) == 0 or int(row['source']) in source) and record['span1_end'] <= record[
                    'span2_start'] and record['span2_end'] < len(tokens) and ''.join(
                    tokens[record['span1_start']:record['span1_end']]) == ''.join(span1_tokens) and ''.join(
                    tokens[record['span2_start']:record['span2_end']]) == ''.join(span2_tokens):
                    records.append(record)
                    records_df.append(row)
                else:
                    excluded_records.append([record, row])
            else:
                excluded_rows.append(row)
        except Exception as e:
            print("error in converting the record. global id: {}. detail: {}".format(row['global_id'], str(e)))
            pass

    # saving records into a JSON file
    if save_json and len(records) > 0:
        with open(str(output_file_name), 'w') as fout:
            json.dump(records, fout)

    return records, records_df, excluded_records, excluded_rows


def brat2crest(input_dir, crest_file_path):
    """
    converting a brat formatted corpus to crest: cresting the corpus!
    :return:
    """

    # first, we need to load the CREST data we converted to BRAT
    df = pd.read_excel(crest_file_path)

    files = os.listdir(input_dir)
    files = [file for file in files if ".ann" in file]

    for file in files:
        global_id = int(file.split('.')[0])
        tags = {}
        with open(input_dir + "/" + file, 'r') as f:
            lines = f.readlines()

        # reading all tags information
        for line in lines:
            line_cols = line.split('\t')
            tags[line_cols[0]] = [line_cols[1].replace('\n', '')]

        # reading context
        context = df.loc[df['global_id'] == global_id].iloc[0]['context']

        # iterate through causal tags
        for key, value in tags.items():
            try:
                if key.startswith("R"):
                    args = value[0].split(' ')
                    arg1_id = args[1].split(':')[1]
                    arg2_id = args[2].split(':')[1]

                    arg1_start = int(tags[arg1_id][0].split(' ')[1])
                    arg1_end = int(tags[arg1_id][0].split(' ')[2])
                    arg2_start = int(tags[arg2_id][0].split(' ')[1])
                    arg2_end = int(tags[arg2_id][0].split(' ')[2])

                    span1 = context[arg1_start:arg1_end]
                    span2 = context[arg2_start:arg2_end]

                    idx_val = {"span1": [[arg1_start, arg1_end]],
                               "span2": [[arg2_start, arg2_end]],
                               "signal": []}

                    row = {"span1": [span1], "span2": [span2], "signal": [],
                           "idx": {"span1": [[arg1_start, arg1_end]], "span2": [[arg2_start, arg2_end]],
                                   "signal": []}, "context": context}

                    # checking if new indexes are correct and align
                    if Converter().check_span_indexes(row):
                        # saving new information to CREST data frame
                        row["idx"] = idx_to_string(row["idx"])
                        row_index = df.loc[df['global_id'] == global_id].index
                        df.at[row_index, 'span1'] = [span1]
                        df.at[row_index, 'span2'] = [span2]
                        df.at[row_index, 'idx'] = idx_to_string(str(idx_val))

            except Exception as e:
                print("[crest-log] Error in converting brat to crest. Detail: {}".format(e))

    # saving the updated df
    df.to_excel(crest_file_path)


def crest2brat(df, output_dir):
    """
    converting a CREST-formatted data frame to BRAT
    :param df: a CREST-formatted data
    :param output_dir: folder were files will be saved
    :return:
    """
    if not type(df) == pd.core.frame.DataFrame:
        print("first parameter should be a pandas data frame")
        raise TypeError

    # first, check if the 'source' directory exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for index, row in df.iterrows():
        ann_file = ""
        t_idx = 1
        args_count = 0
        idx = ast.literal_eval(str(row['idx']))
        span1 = idx['span1']
        span2 = idx['span2']
        signal = idx['signal']
        label = int(row['label'])
        direction = int(row['direction'])

        span1_string = ' '.join(ast.literal_eval(str(row['span1'])))
        span2_string = ' '.join(ast.literal_eval(str(row['span2'])))
        signal_string = ' '.join(ast.literal_eval(str(row['signal'])))

        if len(span1) > 0:
            span_type = 'Span1'
            if label == 1 and direction == 0:
                span_type = 'Cause'
            elif label == 1 and direction == 1:
                span_type = 'Effect'
            ann_file += "T{}\t{} ".format(t_idx, span_type)
            spans1 = []
            for span in span1:
                spans1.append("{} {}".format(span[0], span[1]))
            ann_file += (';'.join(spans1)).strip()
            ann_file += "\t{}\n".format(span1_string)
            t_idx += 1
            args_count += 1

        if len(span2) > 0:
            span_type = 'Span2'
            if label == 1 and direction == 0:
                span_type = 'Effect'
            elif label == 1 and direction == 1:
                span_type = 'Cause'
            ann_file += "T{}\t{} ".format(t_idx, span_type)
            spans2 = []
            for span in span2:
                spans2.append("{} {}".format(span[0], span[1]))
            ann_file += (';'.join(spans2)).strip()
            ann_file += "\t{}\n".format(span2_string)
            t_idx += 1
            args_count += 1

        if len(signal) > 0:
            ann_file += "T{}\tSignal ".format(t_idx)
            signals = []
            for span in signal:
                signals.append("{} {}".format(span[0], span[1]))
            ann_file += (';'.join(signals)).strip()
            ann_file += "\t{}\n".format(signal_string)
            t_idx += 1

        if label == 1:
            if args_count == 2:
                if direction == 0:
                    ann_file += "R1\tCausal Arg1:T1 Arg2:T2\n"
                elif direction == 1:
                    ann_file += "R1\tCausal Arg1:T2 Arg2:T1\n"
        elif label == 0:
            if args_count == 2:
                if direction == 0:
                    ann_file += "R1\tNonCausal Arg1:T1 Arg2:T2\n"
                elif direction == 1:
                    ann_file += "R1\tNonCausal Arg1:T2 Arg2:T1\n"

        ann_file = ann_file.strip('\n')

        # writing .ann and .txt files
        file_name = "{}".format(str(row['global_id']))

        with open('{}/{}.ann'.format(output_dir, file_name), 'w') as file:
            file.write(ann_file)
        with open('{}/{}.txt'.format(output_dir, file_name), 'w') as file:
            file.write(row['context'])
