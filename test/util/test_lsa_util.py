import os
import pickle
import unittest

from util.corpus_util import load_corpus
from util.lsa_util import align


class TestLsaUtil(unittest.TestCase):

    def test_align(self):
        id = '171001'
        ls_corpus = load_corpus(r'E:\librispeech-corpus')
        corpus_entry = ls_corpus[id]
        aligned_vas = align(corpus_entry.audio, corpus_entry.rate, corpus_entry.transcript)
        for va in aligned_vas:
            print(len(va.audio))
            print(va.transcript)
            print(va.alignment_text)

        target_path = os.path.join('../../assets/alignments', id + '.pkl')
        with open(target_path, 'wb') as f:
            pickle.dump(aligned_vas, f)
