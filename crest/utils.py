import os
import ast
import json
import copy
import spacy
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import xml.etree.ElementTree as ET


class DataColumns:
    """
    a class for saving/handling different dataframe column names
    """

    def __init__(self):
        return

    def get_pdtb3_cols(self):
        cols = [
            'RelationType',
            'ConnSpanList',
            'ConnSrc',
            'ConnType',
            'ConnPol',
            'ConnDet',
            'ConnFeatSpanList',
            'Conn1',
            'SClass1A',
            'SClass1B',
            'Conn2',
            'SClass2A',
            'SClass2B',
            'Sup1SpanList',
            'Arg1SpanList',
            'Arg1Src',
            'Arg1Type',
            'Arg1Pol',
            'Arg1Det',
            'Arg1FeatSpanList',
            'Arg2SpanList',
            'Arg2Src',
            'Arg2Type',
            'Arg2Pol',
            'Arg2Det',
            'Arg2FeatSpanList',
            'Sup2SpanList',
            'AdjuReason',
            'AdjuDisagr',
            'PBRole',
            'PBVerb',
            'Offset',
            'Provenance',
            'Link',
            'FullRawText',
            'RelationId'
        ]
        return cols


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
                        record['span1_start'] = i
                        span1_tokens, record['span1_end'] = get_token_indices(i, token_idx, span1_end, tokens)
                    elif token_idx == span2_start:
                        record['span2_start'] = i
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


def brat2crest():
    """
    converting a brat formatted corpus to crest: cresting the corpus!
    :return:
    """
    print("work in progress!")


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

    # first, check if the 'source' directory exist
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
        file_name = "{}_{}_{}_{}".format(str(row['source']), str(label), str(direction), str(row['original_id']))
        # if str(row['ann_file']) != "":
        #    file_name += '_' + str(row['ann_file']).replace('.ann', '').replace('.', '')
        with open('{}/{}.ann'.format(output_dir, file_name), 'w') as file:
            file.write(ann_file)
        with open('{}/{}.txt'.format(output_dir, file_name), 'w') as file:
            file.write(row['context'])


def filter_by_span_length(df, min_len=2, max_len=2):
    """
    getting a subset of relations based on the number of tokens in spans
    :param df: a CREST-formatted data
    :param min_len: minimum total number of tokens in spans of a relation
    :param max_len: maximum total number of tokens in spans of a relation
    :return:
    """
    if min_len > max_len:
        raise Exception('minimum length cannot be larger than maximum length')

    res = pd.DataFrame(columns=list(df))

    for index, row in df.iterrows():
        span1 = ' '.join(ast.literal_eval(row['span1']))
        span2 = ' '.join(ast.literal_eval(row['span2']))

        len_span = 0

        if span1.strip() != "" and span2.strip() != "":
            len_span += len(span1.strip().split(' '))
            len_span += len(span2.strip().split(' '))

        if min_len < len_span <= max_len:
            res = res.append(row)

    return res.reset_index()


def min_avg_max(df):
    """
    finding min, avg., and max length of samples in a CREST-formatted data frame
    :param df:
    :return:
    """
    if not type(df) == pd.core.frame.DataFrame:
        print("first parameter should be a pandas data frame")
        raise TypeError

    max_dict = {'len_max': 0, 'original_id': -1}
    min_dict = {'len_min': 1000000000, 'original_id': -1}  # good for now, but maybe replace it with sys.maxsize later
    context_max = ""
    context_min = ""
    len_total = 0
    for index, row in df.iterrows():
        len_total += len(row.context.split(' '))
        if len(str(row.context).split(' ')) > max_dict['len_max']:
            max_dict['len_max'] = len(str(row.context).split(' '))
            max_dict['original_id'] = row.original_id
            context_max = str(row.context)
        if len(str(row.context).split(' ')) < min_dict['len_min']:
            min_dict['len_min'] = len(str(row.context).split(' '))
            min_dict['original_id'] = row.original_id
            context_min = str(row.context)

    print("Avg. length: {}".format(len_total / len(df)))
    print("+++++++++++++++")
    print("min length/id: {}".format(str(min_dict)))
    print("min context: {}".format(context_min))
    print("+++++++++++++++")
    print("max length/id: {}".format(str(max_dict)))
    print("max context: {}".format(context_max))


def split_statistics(df):
    """
    printing the statistics of a dataframe in terms of number of +/- samples in each split
    :param df:
    :return:
    """
    a = df.loc[df['split'] == 0]
    b = df.loc[df['split'] == 1]
    c = df.loc[df['split'] == 2]

    splits = {'train': a, 'dev': b, 'test': c}

    print('train: {}, dev: {}, test: {}'.format(set(list(a['source'])), set(list(b['source'])), set(list(c['source']))))
    print('train: {}, dev: {}, test: {}'.format(len(a), len(b), len(c)))

    for key, value in splits.items():
        print(key)
        print("+: {}".format(len(value.loc[value['label'] == 1])))
        print("-: {}".format(len(value.loc[value['label'] == 0])))
        print('+++++++++++')


def balance_direction(df, labels=[0, 1]):
    _n_dir = min([len(df[df['direction'] == labels[1]]), len(df[df['direction'] == labels[0]])])

    df_neg = df.loc[df['direction'] == labels[0]].sample(n=_n_dir, random_state=42)
    df_pos = df.loc[df['direction'] == labels[1]].sample(n=_n_dir, random_state=42)

    df = pd.concat([df_neg, df_pos])

    return df.sample(frac=1, random_state=42)


def balance_split(df, classes=[0, 1], balance_dir=False, dir_first=False):
    """
    creating a balanced dataframe of positive and negative samples
    :param df:
    :param balance_dir: if True, also balance the direction
    :param dir_first: if True, balance the direction first
    :param classes: list of two binary labels
    :return:
    """

    if dir_first:
        df = balance_direction(df)

    n_pos = len(df[df['label'] == classes[1]])
    n_neg = len(df[df['label'] == classes[0]])

    _n = min([n_pos, n_neg])

    df_pos = df.loc[df['label'] == classes[1]].sample(n=_n, random_state=42)
    df_neg = df.loc[df['label'] == classes[0]].sample(n=_n, random_state=42)

    if balance_dir:
        df = pd.concat([balance_direction(df_pos), balance_direction(df_neg)])
    else:
        df = pd.concat([df_pos, df_neg])

    # shuffling samples
    df = df.sample(frac=1, random_state=42)

    return df


def create_binary_splits(df, frac_val, classes=[0, 1], resolve_overlap=True):
    """
    creating two splits of a data frame (train/test or train/dev)
    :param df:
    :param frac_val:
    :param classes:
    :param resolve_overlap:
    :return:
    """
    # getting number of samples in each class
    n_neg = len(df[df['label'] == classes[0]])
    n_pos = len(df[df['label'] == classes[1]])

    print("# of samples: positive: {}, negative: {}".format(n_pos, n_neg))

    args = {'frac_val': frac_val, 'n_neg': n_neg, 'n_pos': n_pos}

    # creating splits
    df_neg = df.loc[df['label'] == classes[0]].sample(n=args['n_neg'], random_state=42)
    neg_train = df_neg.apply(lambda x: x.sample(frac=args['frac_val'], random_state=42))
    neg_test = df_neg.drop(neg_train.index)

    df_pos = df.loc[df['label'] == classes[1]].sample(n=args['n_pos'], random_state=42)
    pos_train = df_pos.apply(lambda x: x.sample(frac=args['frac_val'], random_state=42))
    pos_test = df_pos.drop(pos_train.index)

    # concatenating classes' samples
    train_df = pd.concat([neg_train, pos_train])
    test_df = pd.concat([neg_test, pos_test])

    # shuffling samples
    train_df = train_df.sample(frac=1, random_state=42)
    test_df = test_df.sample(frac=1, random_state=42)

    # removing the overlaps
    if resolve_overlap:
        train_df = resolve_context_overlap(train_df, test_df, mode=2)

    # making each split balanced, if needed
    # train_df = balance_split(train_df)
    # test_df = balance_split(test_df)

    return train_df, test_df


def resolve_context_overlap(df1, df2, mode=2):
    """
    removing overlapped context from df1
    :param df1:
    :param df2:
    :param mode: 1: resolve full overlaps, 2: resolve partial overlaps
    :return:
    """

    new_df1 = pd.DataFrame(columns=list(df1))

    df2_context = []
    for index, row in df2.iterrows():
        df2_context.append(str(row['context']).lower().strip())

    if mode == 1:
        for index, row in df1.iterrows():
            if str(row['context']).lower().strip() not in set(df2_context):
                new_df1 = new_df1.append(row)
    elif mode == 2:
        for index, row in df1.iterrows():
            curr_context = str(row['context']).lower().strip()
            if not any(curr_context in x for x in df2_context):
                new_df1 = new_df1.append(row)
    return new_df1


def copa2bert(folder_path, split='dev'):
    """
    converting Choice of Plausible Alternatives (COPA) to BERT's next sentence prediction format
    :param folder_path: path to the COPA folder that contains all .xml files
    :param split: one of the following values: ['dev', 'test']
    :return:
    """

    data = []

    try:
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(folder_path + "copa-{}.xml".format(split), parser=parser)
        root = tree.getroot()

        for item in root.findall("./item"):
            spans = {0: item[0].text, 1: item[1].text, 2: item[2].text}

            span1_premise = spans[0]
            answer = int(item.attrib["most-plausible-alternative"])
            span2_correct = spans[answer]

            span2_incorrect = spans[2] if answer == 1 else spans[1]

            if item.attrib["asks-for"] == "cause":
                data.append({'sent_1': span2_correct, 'sent_2': span1_premise, 'label': 1})
                data.append({'sent_1': span2_incorrect, 'sent_2': span1_premise, 'label': 0})
            elif item.attrib["asks-for"] == "effect":
                data.append({'sent_1': span1_premise, 'sent_2': span2_correct, 'label': 1})
                data.append({'sent_1': span1_premise, 'sent_2': span2_incorrect, 'label': 0})

    except Exception as e:
        print("[crest-log] COPA. Detail: {}".format(e))

    return data
