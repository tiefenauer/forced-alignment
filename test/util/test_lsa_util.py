import pickle
import unittest
from os.path import join

from util.corpus_util import get_corpus
from util.lsa_util import align


class TestLsaUtil(unittest.TestCase):

    def test_align(self):
        id = '171001'
        ls_corpus = get_corpus('ls')
        corpus_entry = ls_corpus[id]
        aligned_vas = align(corpus_entry.audio, corpus_entry.rate, corpus_entry.transcript)
        for va in aligned_vas:
            print(len(va.audio))
            print(va.transcript)
            print(va.alignment_text)

        target_path = join('../../assets/alignments', id + '.pkl')
        with open(target_path, 'wb') as f:
            pickle.dump(aligned_vas, f)
