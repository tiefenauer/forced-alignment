import unittest

from util.corpus_util import get_corpus


class TestCorpusUtil(unittest.TestCase):

    def test_load_corpus(self):
        rl_corpus = get_corpus('rl')
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
        ls_corpus = get_corpus('ls')
        train_set, dev_set, test_set = ls_corpus.train_dev_test_split()
        ls_corpus.summary()
