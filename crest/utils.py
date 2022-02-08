import ast
import pandas as pd


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


def idx_to_string(idx_val):
    """
    converting the dictionary of span indices to string
    :param idx_val: dictionary in form of: {"span1": [], "span2": [], "signal": []}
    :return:
    """
    string = ""
    for key, values in idx_val.items():
        string += key + " "
        for val in values:
            string += str(val[0]) + " " + str(val[1]) + " "
        string = string.strip() + "\n"
    return string.strip()


def string_to_idx(string):
    """
    converting string of span indices to a dictionary in form of {"span1": [], "span2": [], "signal": []}
    :param string: string of span indices in form of:
    span1 start_1 end_1 ... start_n end_n
    span2 start_1 end_1 ... start_n end_n
    signal start_1 end_1 ... start_n end_n
    :return:
    """
    idx_val = {"span1": [], "span2": [], "signal": []}
    string = string.strip().split('\n')
    for index, (key, value) in enumerate(idx_val.items()):
        i = 1
        spans = string[index].split(' ')
        while i < len(spans) - 1:
            idx_val[key].append([int(spans[i]), int(spans[i + 1])])
            i += 2
    return idx_val
