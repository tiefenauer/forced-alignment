from unittest import TestCase

import corpus_util
import webrtc_util


class TestWebRtcUtil(TestCase):

    def test_split_into_segments(self):
        rl_corpus = corpus_util.load_corpus(r'E:\readylingua-corpus\readylingua.corpus')
        corpus_entry = rl_corpus[0]
        webrtc_util.split_segments(corpus_entry)