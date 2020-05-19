import re
import os
import sys
import copy
import spacy
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


class Converter:
    """
    idx = {'span1': [], 'span2': [], 'signal': []} -> indexes of span1, span2, and signal tokens/spans in context
    each value in the idx dictionary is a list of lists of indexes. For example, if span1 has multi tokens in context
    with start:end indexes 2:5 and 10:13, respectively, span1's value in 'idx' will be [[2, 5],[10, 13]]. Lists are
    sorted based on the start indexes of tokens. Same applies for span2 and signal.
    -------------------------------------------------------------------------------
    label = (0 = non-causal), (1 = [span1, span2] -> cause-effect), (2 = [span1, span2] -> effect-cause)
    -------------------------------------------------------------------------------
    split -> 0: train, 1: dev, test: 2. This is the split/part that a relation belongs to in the original dataset.
    For example, if split value for a relation is 1, it means that in the original dataset, the relation is used in the
    development set
    """

    def __init__(self):
        root_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
        sys.path.insert(0, root_path)
        self.dir_path = root_path + "/data/causal/"
        self.scheme_columns = ['original_id', 'span1', 'span2', 'signal', 'context', 'idx', 'label', 'source',
                               'ann_file',
                               'split']

        # loading spaCy's english model (we use spaCy's sentence splitter for long context)
        self.nlp = spacy.load("en_core_web_sm")
        self.semeval_2007_4_code = 1
        self.semeval_2010_8_code = 2
        self.event_causality_code = 3
        self.causal_timebank_code = 4
        self.event_storyline_code = 5
        self.caters_code = 6
        self.because_code = 7

    def convert_semeval_2007_4(self):
        """
        reading SemEval 2007 task 4 data
        :return: pandas data frame of samples
        """
        e1_tag = ["<e1>", "</e1>"]
        e2_tag = ["<e2>", "</e2>"]

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

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                   "signal": []}

                        # replacing tags with standard tags
                        context = context.replace(e1_tag[0], "").replace(e1_tag[1], "").replace(e2_tag[0], "").replace(
                            e2_tag[1], "")

                        samples = samples.append(
                            {"original_id": int(original_id), "span1": [span1], "span2": [span2], "signal": [],
                             "context": context,
                             "idx": idx_val, "label": label, "source": self.semeval_2007_4_code, "ann_file": "",
                             "split": split}, ignore_index=True)

                    except Exception as e:
                        print("[crest-log] semeval07-task4. Detail: {}".format(e))
            return samples

        # reading files
        with open(self.dir_path + 'SemEval2007_task4/task-4-training/relation-1-train.txt', mode='r',
                  encoding='cp1252') as train:
            train_content = train.readlines()

        # this is the test set
        with open(self.dir_path + 'SemEval2007_task4/task-4-scoring/relation-1-score.txt', mode='r',
                  encoding='cp1252') as key:
            test_content = key.readlines()

        data = pd.DataFrame(columns=self.scheme_columns)

        data = data.append(extract_samples(train_content, 0))
        data = data.append(extract_samples(test_content, 2))

        assert self._check_span_indexes(data) == True

        return data

    def convert_semeval_2010_8(self):
        """
        reading SemEval 2010 task 8 data
        :return: pandas data frame of samples
        """
        e1_tag = ["<e1>", "</e1>"]
        e2_tag = ["<e2>", "</e2>"]

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

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                   "signal": []}

                        # replacing tags with standard tags
                        context = context.replace(e1_tag[0], "").replace(e1_tag[1], "").replace(e2_tag[0], "").replace(
                            e2_tag[1], "")

                        samples = samples.append(
                            {"original_id": int(original_id), "span1": [span1], "span2": [span2], "signal": [],
                             "context": context,
                             "idx": idx_val, "label": label, "source": self.semeval_2010_8_code, "ann_file": "",
                             "split": split}, ignore_index=True)

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
                            orig_id = tag.replace('\t', ' ').replace(' ', '_')
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

                        data = data.append(
                            {"original_id": original_id, "span1": [span1], "span2": [span2], "signal": [],
                             "context": context,
                             "idx": idx_val, "label": 1, "source": self.event_causality_code, "ann_file": key,
                             "split": split}, ignore_index=True)
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

                        idx_val = {"span1": [[span1_start, span1_end]], "span2": [[span2_start, span2_end]],
                                   "signal": []}

                        data = data.append(
                            {"original_id": original_id, "span1": [span1], "span2": [span2], "signal": [],
                             "context": context,
                             "idx": idx_val, "label": label, "source": self.event_causality_code, "ann_file": key,
                             "split": split}, ignore_index=True)

        assert self._check_span_indexes(data) == True

        return data

    def convert_causal_timebank(self):
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
                s_event_id = int(link[0].attrib['id'])  # source event id
                t_event_id = int(link[1].attrib['id'])  # target event id

                if 'c-signalID' in link.attrib:
                    signal_id = int(link.attrib['c-signalID'])
                else:
                    signal_id = False

                # finding label direction
                if s_event_id > t_event_id:
                    label = 2
                    s_event_id, t_event_id = t_event_id, s_event_id
                else:
                    label = 1

                context = ""
                span1 = ""
                span2 = ""
                signal = ""
                token_idx = 0

                # finding start and end sentences indexes
                for i in range(len(tokens)):
                    if tokens[i][0] == events[s_event_id][0]:
                        s_sen_id = int(tokens[i][3])
                    if tokens[i][0] == events[t_event_id][0]:
                        t_sen_id = int(tokens[i][3])

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

                data = data.append(
                    {"original_id": original_id, "span1": [span1.strip()], "span2": [span2.strip()],
                     "signal": [signal.strip()],
                     "context": context, "idx": idx_val, "label": label, "source": self.causal_timebank_code,
                     "ann_file": file, "split": ""}, ignore_index=True)

        assert self._check_span_indexes(data) == True

        return data

    def convert_event_storylines(self, version="1.5"):
        """
        converting causal and non-causal samples from EventStoryLines
        """

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
                            if "relType" in relation.attrib:
                                if relation.attrib['relType'] == "PRECONDITION":
                                    label = 1
                                elif relation.attrib['relType'] == "FALLING_ACTION":
                                    label = 2
                                else:
                                    label = 0

                                # --------------------------
                                # building context and spans
                                original_id = relation.attrib["r_id"]
                                source_m_id = int(relation[0].attrib['m_id'])
                                target_m_id = int(relation[1].attrib['m_id'])

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

                                    data = data.append(
                                        {"original_id": original_id, "span1": [span1.strip()], "span2": [span2.strip()],
                                         "signal": [],
                                         "context": context, "idx": idx_val, "label": label,
                                         "source": self.event_storyline_code,
                                         "ann_file": doc, "split": ""}, ignore_index=True)

                                except Exception as e:
                                    print("[crest-log] EventStoryLine. Detail: {}".format(e))

        assert self._check_span_indexes(data) == True

        return data

    def convert_CaTeRS(self):
        folders_path = self.dir_path + "caters/caters_evaluation/"

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

                                        if any(causal_tag in value[0] for causal_tag in causal_tags):
                                            label = 1
                                        else:
                                            label = 0

                                        samples = samples.append(
                                            {"original_id": original_id, "span1": span1[0], "span2": span2[0],
                                             "signal": [],
                                             "context": context,
                                             "idx": idx_val, "label": label, "source": self.caters_code,
                                             "ann_file": doc,
                                             "split": split}, ignore_index=True)

                                except Exception as e:
                                    print("[crest-log] Error in converting CaTeRS. Detail: {}".format(e))
            return samples

        data = pd.DataFrame(columns=self.scheme_columns)
        data = data.append(extract_samples(["dev"], 1))
        data = data.append(extract_samples(["train"], 0))

        assert self._check_span_indexes(data) == True

        return data

    def convert_because(self):
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
                sentences.append([sen.text, sen.start_char])
            return sentences

        def _get_context_spans(doc, arg0_id, arg1_id, signal_id):
            arg0 = ' '.join(tags[arg0_id].split(' ')[1:]).strip()
            if arg1_id != "":
                arg1 = ' '.join(tags[arg1_id].split(' ')[1:]).strip()
                arg1 = arg1.split(';')
            signal = ' '.join(tags[signal_id].split(' ')[1:]).strip()

            arg0 = arg0.split(';')
            signal = signal.split(';')

            df_columns = ['start', 'end', 'tag']
            df = pd.DataFrame(columns=df_columns)

            for arg in arg0:
                df = df.append(pd.DataFrame(
                    [[int(arg.split(' ')[0]), int(arg.split(' ')[1]), "span1"]],
                    columns=df_columns
                ), ignore_index=True)

            if arg1_id != "":
                for arg in arg1:
                    df = df.append(pd.DataFrame(
                        [[int(arg.split(' ')[0]), int(arg.split(' ')[1]), "span2"]],
                        columns=df_columns
                    ), ignore_index=True)

            for arg in signal:
                df = df.append(pd.DataFrame(
                    [[int(arg.split(' ')[0]), int(arg.split(' ')[1]), "signal"]],
                    columns=df_columns
                ), ignore_index=True)

            df = df.sort_values('start')

            doc_segments = _get_doc_sentences(doc)

            i = 0
            while i < len(doc_segments):
                segment_idx = doc_segments[i][1]
                try:
                    if segment_idx <= df.iloc[0]["start"] and (i + 1 == len(doc_segments) or
                                                               (i + 1 < len(doc_segments) and df.iloc[0]["start"] <
                                                                doc_segments[i + 1][1])):
                        context = doc_segments[i][0]
                        span1 = []
                        span2 = []
                        signal = []
                        span1_idxs = []
                        span2_idxs = []
                        signal_idxs = []

                        df_span1 = df.loc[df["tag"] == "span1"]
                        for index, value in df_span1.iterrows():
                            start = value["start"] - segment_idx
                            end = value["end"] - segment_idx
                            span1_idxs.append([start, end])
                            span1.append(context[start:end])

                        df_span2 = df.loc[df["tag"] == "span2"]
                        for index, value in df_span2.iterrows():
                            start = value["start"] - segment_idx
                            end = value["end"] - segment_idx
                            span2_idxs.append([start, end])
                            span2.append(context[start:end])

                        df_signal = df.loc[df["tag"] == "signal"]
                        for index, value in df_signal.iterrows():
                            start = value["start"] - segment_idx
                            end = value["end"] - segment_idx
                            signal_idxs.append([start, end])
                            signal.append(context[start:end])
                        break

                        span1_text = tags[arg0_id][1]
                        if arg1_id != "":
                            span2_text = tags[arg1_id][1]
                        signal_text = tags[signal_id][1]

                        assert (" ".join(span1)).strip() == span1_text, (" ".join(span2)).strip() == span2_text
                        assert (" ".join(signal)).strip() == signal_text
                    i += 1
                except Exception as e:
                    print("[crest-log] detail: {}".format(e))

            return [span1, span1_idxs], [span2, span2_idxs], [signal, signal_idxs], context

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
                            tags[line_cols[0]] = line_cols[1]

                    # reading the corresponding text file for the .ann file
                    with open(docs_path + "/" + doc.strip(".ann") + ".txt", 'r') as f:
                        doc_string = f.read()

                    # now, reading causal relations info
                    for key, value in tags.items():
                        try:
                            original_id = key
                            # check if the relation has all the arguments
                            value = value.replace('\n', '')

                            # causal samples
                            causal_tags = ["Consequence", "Purpose", "Motivation", "NonCausal"]
                            arg_tags = ["Effect", "Cause", "Arg0", "Arg1"]

                            if key.startswith("E") and any(causal_tag in value for causal_tag in causal_tags) and any(
                                    arg_tag in value for arg_tag in arg_tags):
                                args = value.split(' ')
                                signal_tag = args[0].split(':')[1]

                                # check if both arguments are available
                                if len(args) > 2:
                                    arg0_id = args[1].split(':')[1]
                                    arg1_id = args[2].split(':')[1]
                                else:
                                    arg0_id = args[1].split(':')[1]
                                    arg1_id = ""

                                span1, span2, signal, context = _get_context_spans(doc_string, arg0_id, arg1_id,
                                                                                   signal_tag)

                                # specifying the label
                                if "NonCausal" in value:
                                    label = 0
                                else:
                                    if args[1].split(':')[0] == "Cause":
                                        label = 1
                                    else:
                                        label = 2

                                idx_val = {"span1": span1[1],
                                           "span2": span2[1],
                                           "signal": signal[1]}

                                data = data.append(
                                    {"original_id": original_id, "span1": span1[0], "span2": span2[0],
                                     "signal": signal[0],
                                     "context": context,
                                     "idx": idx_val, "label": label, "source": self.because_code,
                                     "ann_file": doc,
                                     "split": ""}, ignore_index=True)

                        except Exception as e:
                            print("[crest-log] {}".format(e))

        assert self._check_span_indexes(data) == True

        return data

    @staticmethod
    def brat2crest():
        """
        converting a brat formatted corpus to crest: cresting the corpus!
        :return:
        """
        print("work in progress!")

    @staticmethod
    def crest2brat():
        """
        converting a crest formatted corpus to brat
        :return:
        """
        print("work in progress!")

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
        checking if spans/signal indexes are correctly stored
        :param data:
        :return:
        """
        for index, row in data.iterrows():
            span1 = ""
            span2 = ""
            signal = ""

            for arg in row["idx"]["span1"]:
                span1 += row["context"][arg[0]:arg[1]] + " "

            for arg in row["idx"]["span2"]:
                span2 += row["context"][arg[0]:arg[1]] + " "

            for sig in row["idx"]["signal"]:
                signal += row["context"][sig[0]:sig[1]] + " "

            if span1.strip() != (" ".join(row["span1"])).strip() or span2.strip() != (
                    " ".join(row["span2"])).strip() or signal.strip() != (" ".join(row["signal"])).strip():
                print("span1 -> {}:{}".format(span1, (" ".join(row["span1"])).strip()))
                print("span2 -> {}:{}".format(span2, (" ".join(row["span2"])).strip()))
                print("signal -> {}:{}".format(signal, (" ".join(row["signal"])).strip()))
                return False
        return True
