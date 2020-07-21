import ast
import json

import pandas as pd

from os import path


def crest2tacred(df, output_file_name, split=[], save_json=False):
    """
    converting CREST-formatted data to TACRED (https://nlp.stanford.edu/projects/tacred/)
    :param df: pandas data frame of the CREST-formatted excel file
    :param output_file_name: name of output file without extension
    :param save_json: binary value, True, if want to save result in a JSON file, False, otherwise
    :param split: split of the data, value is a list of numbers such as 0: train, 1: dev, test: 2. will return all data by default
    :return: list of dictionaries
    """
    if not type(df) == pd.core.frame.DataFrame:
        print("first parameter should be a pandas data frame")
        raise TypeError

    records = []
    for index, row in df.iterrows():
        try:
            idx = ast.literal_eval(row['idx'])
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
                        span1_tokens = ast.literal_eval(row['span1'])[0].split(' ')
                        record['span1_end'] = i + len(span1_tokens) - 1
                    elif token_idx == span2_start:
                        record['span2_start'] = i
                        span2_tokens = ast.literal_eval(row['span2'])[0].split(' ')
                        record['span2_end'] = i + len(span2_tokens) - 1
                    token_idx += len(tokens[i]) + 1

                # getting the label (ignoring the relation direction for now)
                if int(row['label']) > 0:
                    label = 1
                else:
                    label = 0

                record['id'] = str(row['original_id']) + str(row['source'])
                record['token'] = tokens
                record['relation'] = label
                features = ['id', 'token', 'span1_start', 'span1_end', 'span2_start', 'span2_end', 'relation']
                # check if record has all the required fields
                if all(feature in record for feature in features) and (len(split) == 0 or int(row['split']) in split):
                    records.append(record)
        except Exception as e:
            print("error in converting the record. id: {}-{}. detail: {}".format(row['original_id'], row['source'],
                                                                                 str(e)))
            pass

    # saving records into a JSON file
    if save_json and len(records) > 0:
        with open('../data/causal/{}.json'.format(str(output_file_name)), 'w') as fout:
            json.dump(records, fout)

    return records
