from unittest import TestCase

import corpus_util
import webrtc_util


class TestWebRtcUtil(TestCase):

    def test_split_into_segments(self):
        rl_corpus = corpus_util.load_corpus(r'E:\readylingua-corpus\readylingua.corpus')
        corpus_entry = rl_corpus[0]
        webrtc_util.split_into_segments(corpus_entry)

    def test_calculate_boundaries(self):
        rl_corpus = corpus_util.load_corpus(r'E:\readylingua-corpus\readylingua.corpus')
        corpus_entry = rl_corpus[0]
        webrtc_util.calculate_boundaries(corpus_entry)
