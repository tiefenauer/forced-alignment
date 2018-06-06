import os
from unittest import TestCase

import numpy as np

from util import corpus_util, webrtc_util
from src.util.audio_util import write_wav_file


class TestWebRtcUtil(TestCase):

    def test_split_into_segments(self):
        rl_corpus = corpus_util.load_corpus(r'E:\readylingua-corpus')
        corpus_entry = rl_corpus[0]

        voiced_segments, unvoiced_segments = webrtc_util.split_segments(corpus_entry)
        for i, voiced_segment in enumerate(voiced_segments):
            audio = np.concatenate([frame.audio for frame in voiced_segment])
            file_path = os.path.join('..', f'chunk-{i:0002d}.wav')
            print(f'Writing {file_path}')
            write_wav_file(file_path, corpus_entry.rate, audio)

        corpus_entry = rl_corpus[0]
        webrtc_util.split_segments(corpus_entry)
