from unittest import TestCase

import data_util


class TestDataUtil(TestCase):

    def test_load(self):
        train_set = data_util.load_subset('train', r'E:\librispeech-data')
        print(len(list(train_set)))
        dev_set = data_util.load_subset('dev', r'E:\librispeech-data')
        print(len(list(dev_set)))
        test_set = data_util.load_subset('test', r'E:\librispeech-data')
        print(len(list(test_set)))
