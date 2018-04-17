import unittest

import corpus_util


class CorpusUtilTest(unittest.TestCase):

    def test_load_corpus(self):
        corpus = corpus_util.load_corpus('readylingua/readylingua.corpus')
        print(len(corpus))
