from os.path import join
from unittest import TestCase

import numpy as np
from librosa.output import write_wav

from util import vad_util
from util.corpus_util import get_corpus


class TestWebRtcUtil(TestCase):

    def test_split_into_segments(self):
        rl_corpus = get_corpus('rl')
        corpus_entry = rl_corpus[0]

        audio, rate = corpus_entry.audio, corpus_entry.rate

        voiced_segments, unvoiced_segments = vad_util.webrtc_split(audio, rate)
        for i, voiced_segment in enumerate(voiced_segments):
            audio = np.concatenate([frame.audio for frame in voiced_segment])
            file_path = join('..', f'chunk-{i:0002d}.wav')
            print(f'Writing {file_path}')
            write_wav(file_path, audio, corpus_entry.rate)

    def test_split_into_segments_en(self):
        ls_corpus = get_corpus('ls')
        corpus_entry = ls_corpus['171001']

        audio, rate = corpus_entry.audio, corpus_entry.rate

        voiced_segments, unvoiced_segments = vad_util.webrtc_split(audio, rate)
        for i, voiced_segment in enumerate(voiced_segments):
            audio = np.concatenate([frame.audio for frame in voiced_segment])
            file_path = join('..', f'chunk-{i:0002d}.wav')
            print(f'Writing {file_path}')
            write_wav(file_path, audio, corpus_entry.rate)
