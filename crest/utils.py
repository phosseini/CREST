import ast
import json

import pandas as pd

from os import path


def crest2tacred(input_path, save_json=False):
    """
    converting CREST-formatted data to TACRED (https://nlp.stanford.edu/projects/tacred/)
    :param input_path: path to the CREST-formatted excel file
    :param save_json: binary value, True, if want to save result in a JSON file, False, otherwise
    :return: list of dictionaries
    """

    # checking the input file path
    if not path.isfile(input_path):
        print("file {} does not exist".format(input_path))
        raise FileNotFoundError

    # reading CREST data
    df = pd.read_excel(input_path, index_col=[0])

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
                records.append(record)
        except Exception as e:
            print("error in converting the record. id: {}-{}. detail: {}".format(row['original_id'], row['source'],
                                                                                 str(e)))
            pass

    # saving records into a JSON file
    if save_json and len(records) > 0:
        with open('../data/causal/crest_tacred.json', 'w') as fout:
            json.dump(records, fout)

    return records
