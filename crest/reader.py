class Reader:
    def __init__(self):
        pass

    @staticmethod
    def get_text_between(row):
        """
        reading text between span1 and span2
        :param row: a crest formatted row
        :return:
        """
        span1_idx = row["idx"]["span1"]
        span2_idx = row["idx"]["span2"]
        # check if we have both spans, otherwise, there is no text between
        if len(span1_idx) > 0 and len(span2_idx):
            span1_start = span1_idx[0][0]
            span2_start = span2_idx[0][0]

            if span2_start < span1_start:
                span1_idx, span2_idx = span2_idx, span1_idx

            between_start = span1_idx[len(span1_idx) - 1][1]
            between_end = span2_idx[0][0]

            # double check the indexes in cases where we have spans with overlaps and multiple tokens
            if between_end > between_start:
                return row["context"][between_start:between_end]
