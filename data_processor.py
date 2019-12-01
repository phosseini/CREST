import re

from data_reader import CausalDataReader


class CausalDataProcessor:
    """
    helper class for our causal relation schema
    """
    # TODO: handling those cases that an argument may have multiple parts.
    def __init__(self):
        self.data_reader = CausalDataReader()

    def remove_text_tags(self, text):
        text = text.replace(self.data_reader.signal_tag[0], "")
        text = text.replace(self.data_reader.signal_tag[1], "")
        text = text.replace(self.data_reader.arg1_tag[0], "")
        text = text.replace(self.data_reader.arg1_tag[1], "")
        text = text.replace(self.data_reader.arg2_tag[0], "")
        text = text.replace(self.data_reader.arg2_tag[1], "")
        return text.strip().lower()

    @staticmethod
    def get_tag_text(tag, text):
        """
        get text span of a tag (if there's multiple occurrences, we return the first occurrence)
        :param tag: a list of opening and closing tags
        :param text: tagged sentence/text
        :return:
        """
        result = re.findall(tag[0] + "(.*?)" + tag[1], text)
        if result is not None and len(result) > 0:
            return result[0].strip()
        return ""

    def get_text_between(self, x, arg1_tag=[], arg2_tag=[]):
        """
        this method gets the text between two arguments of a causal relation
        :param x: the text field of a record with tags in the causal relation data frame
        :param arg1_tag: by default, the first tag in our causal scheme
        :param arg2_tag: by default, the second tag in our causal schema
        :return:
        """

        def _clean_text(text):
            # TODO: ideally we should only be removing signal tags.
            text = text.replace(self.data_reader.signal_tag[0], "")
            text = text.replace(self.data_reader.signal_tag[1], "")
            text = text.replace(self.data_reader.arg1_tag[0], "")
            text = text.replace(self.data_reader.arg1_tag[1], "")
            text = text.replace(self.data_reader.arg2_tag[0], "")
            text = text.replace(self.data_reader.arg2_tag[1], "")
            return text

        if arg1_tag == [] and arg2_tag == []:
            arg1_tag = self.data_reader.arg1_tag
            arg2_tag = self.data_reader.arg2_tag

        try:
            # in cases that we have <arg1></arg1> . . . <arg2></arg2>
            result = re.search(arg1_tag[1] + "(.*)" + arg2_tag[0], x)
            if result is None:
                # in cases that we have <arg2></arg2> . . . <arg1></arg1>
                result = re.search(arg2_tag[1] + "(.*)" + arg1_tag[0], x)
                if result is None:
                    return ""
                else:
                    return _clean_text(result.group(1))
            else:
                return _clean_text(result.group(1))
        except Exception as e:
            print("[log-preprocessing-text-between] detail: " + str(e) + ". text: " + str(x))


    def get_text_left(self, x):
        """
        this method gets the text of the left side of the first argument in a causal relation
        :param x: the text field of a record with tags in the causal relation data frame
        :return:
        """

        arg1_tag = self.data_reader.arg1_tag
        arg2_tag = self.data_reader.arg2_tag

        # in cases that we have <arg1></arg1> . . . <arg2></arg2>
        result = re.search(arg1_tag[1] + "(.*)" + arg2_tag[0], x)
        if result is None:
            # in cases that we have <arg2></arg2> . . . <arg1></arg1>
            result = re.search(arg2_tag[1] + "(.*)" + arg1_tag[0], x)
            if result is not None:
                result = re.search("(.*)" + arg2_tag[0], x)
                if result is not None:
                    return result.group(1)
        else:
            result = re.search("(.*)" + arg1_tag[0], x)
            if result is not None:
                return result.group(1)
        return ""

    def get_text_right(self, x):
        """
        this method gets the text of the left side of the first argument in a causal relation
        :param x: the text field of a record with tags in the causal relation data frame
        :return:
        """

        arg1_tag = self.data_reader.arg1_tag
        arg2_tag = self.data_reader.arg2_tag

        # in cases that we have <arg1></arg1> . . . <arg2></arg2>
        result = re.search(arg1_tag[1] + "(.*)" + arg2_tag[0], x)
        if result is None:
            # in cases that we have <arg2></arg2> . . . <arg1></arg1>
            result = re.search(arg2_tag[1] + "(.*)" + arg1_tag[0], x)
            if result is not None:
                result = re.search(arg1_tag[1] + "(.*)", x)
                if result is not None:
                    return result.group(1)
        else:
            result = re.search(arg2_tag[1] + "(.*)", x)
            if result is not None:
                return result.group(1)
        return ""