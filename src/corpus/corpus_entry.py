from copy import deepcopy
from datetime import timedelta
from random import randint

import numpy as np
from os.path import exists, join
from tabulate import tabulate

from corpus.audible import Audible
from util.audio_util import read_wav_file
from util.string_util import contains_numeric


class CorpusEntry(Audible):

    def __init__(self, audio_file, segments, original_path='', parms={}):
        self.corpus = None
        self.audio_file = audio_file

        for segment in segments:
            segment.corpus_entry = self
        self.segments = segments

        self.original_path = original_path
        self.name = parms['name'] if 'name' in parms else ''
        self.id = parms['id'] if 'id' in parms else str(randint(1, 999999))
        self.language = parms['language'] if 'language' in parms else 'N/A'
        self.chapter_id = parms['chapter_id'] if 'chapter_id' in parms else 'N/A'
        self.speaker_id = parms['speaker_id'] if 'speaker_id' in parms else 'N/A'
        self.original_sampling_rate = parms['rate'] if 'rate' in parms else 'N/A'
        self.original_channels = parms['channels'] if 'channels' in parms else 'N/A'
        self.subset = parms['subset'] if 'subset' in parms else 'N/A'
        self.media_info = parms['media_info'] if 'media_info' in parms else {}

    def __iter__(self):
        for segment in self.segments:
            yield segment

    def __getitem__(self, item):
        return self.speech_segments[item]

    def _create_audio_and_rate(self):
        return read_wav_file(self.audio_file)

    @property
    def speech_segments(self):
        return [segment for segment in self.segments if segment.segment_type == 'speech']

    @property
    def speech_segments_unaligned(self):
        return [segment for segment in self.segments if segment.segment_type == 'speech*']

    @property
    def pause_segments(self):
        return [segment for segment in self.segments if segment.segment_type == 'pause']

    @property
    def speech_segments_numeric(self):
        return [segment for segment in self.speech_segments if contains_numeric(segment.text)]

    @property
    def speech_segments_not_numeric(self):
        return [segment for segment in self.speech_segments if not contains_numeric(segment.text)]

    @property
    def transcript(self):
        return '\n'.join(segment.transcript for segment in self.speech_segments)

    @property
    def text(self):
        return '\n'.join(segment.text for segment in self.speech_segments)

    @property
    def x_path(self):
        return join(self.corpus.root_path, self.id + '.X.npy')

    @property
    def y_path(self):
        return join(self.corpus.root_path, self.id + '.Y.npy')

    @property
    def labels(self):
        if exists(self.y_path):
            labels = np.load(self.y_path)
            return labels
        return None

    @property
    def audio_length(self):
        return float(self.media_info['duration'])

    def __getstate__(self):
        # prevent caches from being pickled
        state = dict(self.__dict__)
        if '_audio' in state: del state['_audio']
        if '_rate' in state: del state['_rate']
        return state

    def __call__(self, *args, **kwargs):
        if not kwargs or 'include_numeric' not in kwargs or kwargs['include_numeric'] is True:
            return self
        _copy = deepcopy(self)
        segments = self.speech_segments_not_numeric
        _copy.segments = segments
        _copy.name = self.name + f' ==> only segments without numeric values'
        return _copy

    def summary(self):
        print('')
        print('Corpus Entry: '.ljust(30) + f'{self.name} (id={self.id})')
        print('Audio: '.ljust(30) + self.audio_file)
        print('Spectrogram: '.ljust(30) + self.x_path)
        print('Labels: '.ljust(30) + self.y_path)
        print('')
        l_sg = sum(seg.audio_length for seg in self.speech_segments)
        l_sp = sum(seg.audio_length for seg in self.speech_segments)
        l_ps = sum(seg.audio_length for seg in self.pause_segments)
        l_sp_u = sum(seg.audio_length for seg in self.speech_segments_unaligned)
        l_sp_num = sum(seg.audio_length for seg in self.speech_segments_numeric)
        l_sp_nnum = sum(seg.audio_length for seg in self.speech_segments_not_numeric)
        table = {
            '#speech segments': (len(self.speech_segments), timedelta(seconds=l_sp)),
            '#pause segments': (len(self.pause_segments), timedelta(seconds=l_ps)),
            '#segments (unaligned)': (len(self.speech_segments_unaligned), timedelta(seconds=l_sp_u)),
            '#speech segments containing numbers in transcript': (
                len(self.speech_segments_numeric), timedelta(seconds=l_sp_num)),
            '#speech segments not containing numbers in transcript': (
                len(self.speech_segments_not_numeric), timedelta(seconds=l_sp_nnum)),
            '#total segments': (len(self.segments), timedelta(seconds=l_sg)),
        }
        headers = ['# ', 'hh:mm:ss']
        print(tabulate([(k,) + v for k, v in table.items()], headers=headers))
        print('')
        print(f'duration: {timedelta(seconds=self.audio_length)}')
        print(f'original path: {self.original_path}')
        print(f'original sampling rate: {self.original_sampling_rate}')
        print(f'original #channels: {self.original_channels}')
        print(f'language: {self.language}')
        print(f'chapter ID: {self.chapter_id}')
        print(f'speaker_ID: {self.speaker_id}')
        print(f'subset membership: {self.subset}')
        print(f'media info: {self.media_info}')
