import unittest

from util import corpus_util


class TestCorpusUtil(unittest.TestCase):

    def test_load_corpus(self):
        rl_corpus = corpus_util.load_corpus(r'E:\readylingua-corpus\readylingua.corpus')
        rl_corpus.summary()
        rl_corpus_de = rl_corpus('de')
        rl_corpus_de.summary()

        corpus_entry = rl_corpus[0]
        corpus_entry.summary()
        freq, times, spec = corpus_entry.spectrogram
        print(freq.shape)
        print(times.shape)
        print(spec.shape)
        labels = corpus_entry.labels
        print(labels.shape)
        corpus_entry = rl_corpus['news170524']
        corpus_entry.summary()

        corpus_entry_numeric = corpus_entry(numeric=True)
        corpus_entry_numeric.summary()
        corpus_entry_not_numeric = corpus_entry(numeric=False)
        corpus_entry_not_numeric.summary()
        ls_corpus = corpus_util.load_corpus(r'E:\librispeech-corpus\librispeech.corpus')
        train_set, dev_set, test_set = ls_corpus.train_dev_test_split()
        ls_corpus.summary()
