from unittest import TestCase

import data_util
from corpus_util import load_corpus


class TestDataUtil(TestCase):

    def test_load_subset(self):
        train_set = data_util.load_subset('train', r'E:\librispeech-data')
        print(len(list(train_set)))
        dev_set = data_util.load_subset('dev', r'E:\librispeech-data')
        print(len(list(dev_set)))
        test_set = data_util.load_subset('test', r'E:\librispeech-data')
        print(len(list(test_set)))

    def test_load_labelled_data(self):
        rl_corpus = load_corpus(r'E:\readylingua-corpus\readylingua.corpus')
        root_path = r'E:\readylingua-data'
        corpus_entry = rl_corpus[0]
        freqs, times, spec, subset_name = data_util.load_x(corpus_entry, root_path)
        y, subset_name = data_util.load_y(corpus_entry, root_path)
        print(freqs.shape)
        print(times.shape)
        print(spec.shape)
        print(y.shape)
        print(subset_name)
