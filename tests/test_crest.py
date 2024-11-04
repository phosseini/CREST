import os
import sys

path = os.getcwd()
sys.path.append('{}/src'.format('/'.join(path.split('/')[:-1])))

import unittest
from crest import Converter


class TestCREST(unittest.TestCase):
    converter = Converter()

    def test_converter(self):
        df, mis = self.converter.convert2crest(dataset_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9], save_file=True)

        print("samples: " + str(len(df)))
        print("+ causal: {}".format(len(df.loc[df["label"] == 1])))
        print("- non-causal: {}".format(len(df.loc[df["label"] == 0])))
        print("train: {}".format(len(df.loc[df["split"] == 0])))
        print("dev: {}".format(len(df.loc[df["split"] == 1])))
        print("test: {}".format(len(df.loc[df["split"] == 2])))


if __name__ == '__main__':
    unittest.main()
