class Reader:
    def __init__(self):
        pass

    @staticmethod
    def get_left_between_right(row):
        """
        reading text between span1 and span2, left of span1, and right of span2
        :param row: a crest formatted row
        :return: a list containing [left, between, right] text spans
        """
        span1_idx = row["idx"]["span1"]
        span2_idx = row["idx"]["span2"]

        text_between = False
        text_right = False
        text_left = False

        # TODO: decide if we want to also handle one-span cases

        # check if we have both spans
        if len(span1_idx) > 0 and len(span2_idx) > 0:
            span1_start = span1_idx[0][0]
            span2_start = span2_idx[0][0]

            # making sure span1_start is always smaller
            if span2_start < span1_start:
                span1_idx, span2_idx = span2_idx, span1_idx

            right_start = span2_idx[-1][1]
            right_end = len(row["context"])

            left_start = 0
            left_end = span1_idx[0][0]

            between_start = span1_idx[-1][1]
            between_end = span2_idx[0][0]

            # double check the indexes in cases where we have spans with overlaps and multiple tokens
            if between_end > between_start:
                text_between = row["context"][between_start:between_end]
            if right_end > right_start:
                text_right = row["context"][right_start:right_end]
            if left_end > left_start:
                text_left = row["context"][left_start:left_end]

        return [text_left, text_between, text_right]
