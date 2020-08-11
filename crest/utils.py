import os
import ast
import json
import pandas as pd


def crest2tacred(df, output_file_name, split=[], source=[], save_json=False):
    """
    converting CREST-formatted data to TACRED (https://nlp.stanford.edu/projects/tacred/)
    :param df: pandas data frame of the CREST-formatted excel file
    :param output_file_name: name of output file without extension
    :param save_json: binary value, True, if want to save result in a JSON file, False, otherwise
    :param split: split of the data, value is a list of numbers such as 0: train, 1: dev, test: 2. will return all data by default
    :param source: source of the data, a list of integer numbers
    :return: list of dictionaries
    """
    if not type(df) == pd.core.frame.DataFrame:
        print("first parameter should be a pandas data frame")
        raise TypeError

    records = []
    records_df = []
    for index, row in df.iterrows():
        try:
            idx = ast.literal_eval(str(row['idx']))
            # making sure spans are made of consecutive tokens
            if len(idx['span1']) == 1 and len(idx['span2']) == 1:
                record = {}
                span1_start = idx['span1'][0][0]
                span2_start = idx['span2'][0][0]
                tokens = row['context'].split(' ')

                # creating list of tokens in context and finding spans' start and end indices
                token_idx = 0
                for i in range(len(tokens)):
                    if token_idx == span1_start:
                        record['span1_start'] = i
                        span1_tokens = ast.literal_eval(str(row['span1']))[0].split(' ')
                        record['span1_end'] = i + len(span1_tokens) - 1
                    elif token_idx == span2_start:
                        record['span2_start'] = i
                        span2_tokens = ast.literal_eval(str(row['span2']))[0].split(' ')
                        record['span2_end'] = i + len(span2_tokens) - 1
                    token_idx += len(tokens[i]) + 1

                # getting the label and span type
                if int(row['label']) == 0:
                    label = 0
                    record['span1_type'] = 'O'
                    record['span2_type'] = 'O'
                else:
                    if int(row['label']) == 1:
                        record['span1_type'] = 'S-CAUSE'
                        record['span2_type'] = 'S-EFFECT'
                    elif int(row['label']) == 2:
                        record['span1_type'] = 'S-EFFECT'
                        record['span2_type'] = 'S-CAUSE'
                    label = 1

                record['id'] = str(row['original_id']) + str(row['source'])
                record['token'] = tokens
                record['relation'] = label
                features = ['id', 'token', 'span1_start', 'span1_end', 'span2_start', 'span2_end', 'relation']
                # check if record has all the required fields
                if all(feature in record for feature in features) and (
                        len(split) == 0 or int(row['split']) in split) and (
                        len(source) == 0 or int(row['source']) in source):
                    records.append(record)
                    records_df.append(row)
        except Exception as e:
            print("error in converting the record. id: {}-{}. detail: {}".format(row['original_id'], row['source'],
                                                                                 str(e)))
            pass

    # saving records into a JSON file
    if save_json and len(records) > 0:
        with open(str(output_file_name), 'w') as fout:
            json.dump(records, fout)

    return records, records_df


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

        span1_string = ' '.join(ast.literal_eval(str(row['span1'])))
        span2_string = ' '.join(ast.literal_eval(str(row['span2'])))
        signal_string = ' '.join(ast.literal_eval(str(row['signal'])))

        if len(span1) > 0:
            span_type = 'Span'
            if row['label'] == 1:
                span_type = 'Cause'
            elif row['label'] == 2:
                span_type = 'Effect'
            ann_file += "T{}\t{} ".format(t_idx, span_type)
            spans1 = []
            for span in span1:
                spans1.append("{} {}".format(span[0], span[1]))
            ann_file += (';'.join(spans1)).strip()
            ann_file += "\t{}\n".format(span1_string)
            arg0 = "T{}".format(t_idx)
            t_idx += 1
            args_count += 1

        if len(span2) > 0:
            span_type = 'Span'
            if row['label'] == 1:
                span_type = 'Effect'
            elif row['label'] == 2:
                span_type = 'Cause'
            ann_file += "T{}\t{} ".format(t_idx, span_type)
            spans2 = []
            for span in span2:
                spans2.append("{} {}".format(span[0], span[1]))
            ann_file += (';'.join(spans2)).strip()
            ann_file += "\t{}\n".format(span2_string)
            arg0 = "T{}".format(t_idx)
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

        if row['label'] in [1, 2]:
            if args_count == 1:
                if row['label'] == 1:
                    ann_file += "R1\tCausal Arg1:{}\n".format(arg0)
                elif row['label'] == 2:
                    ann_file += "R1\tCausal Arg2:{}\n".format(arg0)
            elif args_count == 2:
                if row['label'] == 1:
                    ann_file += "R1\tCausal Arg1:T1 Arg2:T2\n"
                elif row['label'] == 2:
                    ann_file += "R1\tCausal Arg1:T2 Arg2:T1\n"
        else:
            if args_count == 1:
                ann_file += "R1\tNonCausal Arg1:{}\n".format(arg0)
            else:
                ann_file += "R1\tNonCausal Arg1:T1 Arg2:T2\n"

        ann_file = ann_file.strip('\n')

        # writing .ann and .txt files
        file_name = "{}_{}_{}".format(str(row['source']), str(row['label']), str(row['original_id']))
        if str(row['ann_file']) != "":
            file_name += '_' + str(row['ann_file']).replace('.ann', '').replace('.', '')
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
