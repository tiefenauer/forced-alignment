import unittest

import corpus_util


class CorpusUtilTest(unittest.TestCase):

    def test_load_corpus(self):
        rl_corpus = corpus_util.load_corpus(r'E:\readylingua-corpus\readylingua.corpus')
        print(len(rl_corpus))
        ls_corpus = corpus_util.load_corpus(r'E:\librispeech-corpus\librispeech.corpus')
        train_set, dev_set, test_set = ls_corpus.train_dev_test_split()
        print(len(ls_corpus))
