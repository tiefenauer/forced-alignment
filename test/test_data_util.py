from unittest import TestCase

import data_util


class TestDataUtil(TestCase):

    def test_load(self):
        X_train, Y_train = data_util.load_subset('train', 'ls', r'E:\librispeech-data')
        print(len(X_train), len(Y_train))
        X_dev, Y_dev = data_util.load_subset('dev', 'ls', r'E:\librispeech-data')
        print(len(X_dev), len(Y_dev))
        X_test, Y_test = data_util.load_subset('test', 'ls', r'E:\librispeech-data')
        print(len(X_test), len(Y_test))
