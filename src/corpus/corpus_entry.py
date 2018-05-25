from copy import deepcopy
from random import randint

import numpy as np

from util.audio_util import read_wav_file
from util.string_util import contains_numeric


class CorpusEntry(object):
    # cache values
    _audio = None
    _rate = None

    def __init__(self, audio_file, segments, original_path='', parms={}):
        self.corpus = None
        self.audio_file = audio_file
        self.x_path = None  # will be set in 02_create_labelled_data.py
        self.y_path = None  # will be set in 02_create_labelled_data.py

        for segment in segments:
            segment.corpus_entry = self
        self.segments = segments

        self.original_path = original_path
        self.name = parms['name'] if 'name' in parms else ''
        self.id = parms['id'] if 'id' in parms else str(randint(1, 999999))
        self.language = parms['language'] if 'language' in parms else 'unknown'
        self.chapter_id = parms['chapter_id'] if 'chapter_id' in parms else 'unknown'
        self.speaker_id = parms['speaker_id'] if 'speaker_id' in parms else 'unkown'
        self.original_sampling_rate = parms['rate'] if 'rate' in parms else 'unknown'
        self.original_channels = parms['channels'] if 'channels' in parms else 'unknown'
        self.subset = parms['subset'] if 'subset' in parms else 'unknown'
        self.media_info = parms['media_info'] if 'media_info' in parms else {}

    def __iter__(self):
        for segment in self.segments:
            yield segment

    def __getitem__(self, item):
        return self.speech_segments[item]

    @property
    def speech_segments(self):
        return [segment for segment in self.segments if segment.segment_type == 'speech']

    @property
    def speech_segments_unaligned(self):
        return [segment for segment in self.segments if segment.segment_type == 'speech*']

    @property
    def speech_segments_numeric(self):
        return [segment for segment in self.segments if contains_numeric(segment.text)]

    @property
    def speech_segments_not_numeric(self):
        return [segment for segment in self.segments if not contains_numeric(segment.text)]

    @property
    def pause_segments(self):
        return [segment for segment in self.segments if segment.segment_type == 'pause']

    @property
    def audio(self):
        if self._audio is None:
            self._rate, self._audio = read_wav_file(self.audio_file)
        return self._rate, self._audio

    @property
    def transcription(self):
        return '\n'.join(segment.transcription for segment in self.speech_segments)

    @property
    def text(self):
        return '\n'.join(segment.text for segment in self.speech_segments)

    @property
    def spectrogram(self):
        if self.x_path:
            (freqs, times, spec) = np.load(self.x_path)
            return freqs, times, spec
        return None

    @property
    def labels(self):
        if self.y_path:
            labels = np.load(self.y_path)
            return labels
        return None

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
        print(f'Corpus Entry: {self.name} (id={self.id})')
        print('-----------------------------------------------------------')
        print(f'# speech segments: {len(self.speech_segments)}')
        print(f'# pause segments: {len(self.pause_segments)}')
        print(f'# total segments: {len(self.segments)}')
        print(f'# unaligned speech segments: {len(self.speech_segments_unaligned)}')
        print(f'# speech segments with numbers: {len(self.speech_segments_numeric)}')
        print(f'# speech segments without numbers: {len(self.speech_segments_not_numeric)}')
        print(f'original path: {self.original_path}')
        print(f'original sampling rate: {self.original_sampling_rate}')
        print(f'original #channels: {self.original_channels}')
        print(f'language: {self.language}')
        print(f'chapter ID: {self.chapter_id}')
        print(f'speaker_ID: {self.speaker_id}')
        print(f'subset membership: {self.subset}')
        print(f'media info: {self.media_info}')