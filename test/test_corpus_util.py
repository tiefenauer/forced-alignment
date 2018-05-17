import unittest

import corpus_util


class CorpusUtilTest(unittest.TestCase):

    def test_load_corpus(self):
        rl_corpus = corpus_util.load_corpus(r'E:\readylingua-corpus\readylingua.corpus')
        rl_corpus.summary()
        corpus_entry = rl_corpus['news170524']
        rl_corpus_de = rl_corpus('de')
        rl_corpus_de.summary()
        ls_corpus = corpus_util.load_corpus(r'E:\librispeech-corpus\librispeech.corpus')
        train_set, dev_set, test_set = ls_corpus.train_dev_test_split()
        ls_corpus.summary()
