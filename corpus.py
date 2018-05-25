from abc import ABC, abstractmethod
from copy import copy, deepcopy
from random import randint

from tqdm import tqdm

from audio_util import read_wav_file
from corpus_util import filter_corpus_entry_by_subset_prefix
from string_utils import normalize, contains_numeric


def calculate_crop(segments):
    crop_start = min(segment.start_frame for segment in segments)
    crop_end = max(segment.end_frame for segment in segments)
    return crop_start, crop_end


def crop_segments(segments):
    cropped_segments = []
    crop_start, crop_end = calculate_crop(segments)
    for segment in segments:
        cropped_segment = copy(segment)
        cropped_segment.start_frame -= crop_start
        cropped_segment.end_frame -= crop_start
        cropped_segments.append(cropped_segment)
    return cropped_segments


class Corpus(ABC):

    def __init__(self, name, corpus_entries, root_path):
        self._name = ''
        self.name = name
        for corpus_entry in corpus_entries:
            corpus_entry.corpus = self
        self.corpus_entries = corpus_entries
        self.root_path = root_path

    def __iter__(self):
        for corpus_entry in self.corpus_entries:
            yield corpus_entry

    def __getitem__(self, val):
        # access by index
        if isinstance(val, int) or isinstance(val, slice):
            return self.corpus_entries[val]
        # access by id
        if isinstance(val, str):
            return next(iter([corpus_entry for corpus_entry in self.corpus_entries if corpus_entry.id == val]), None)
        return None

    def __len__(self):
        return len(self.corpus_entries)

    def __call__(self, *args, **kwargs):
        languages = kwargs['languages'] if 'languages' in kwargs else self.languages
        include_numeric = kwargs['include_numeric'] if 'include_numeric' in kwargs else True
        print(f'filtering languages={languages}')
        entries = [entry for entry in self.corpus_entries if entry.language in languages]
        print(f'found {len(entries)} entries for languages {languages}')

        if not include_numeric:
            print(f'filtering out speech segments with numbers in transcription')
            entries = [entry(include_numeric=include_numeric) for entry in tqdm(entries, unit=' entries')]

        _copy = deepcopy(self)
        _copy.corpus_entries = entries
        return _copy

    @property
    def name(self):
        languages = ', '.join(self.languages)
        return self._name + f' (languages: {languages})'

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def languages(self):
        return set(lang for lang in (corpus_entry.language for corpus_entry in self.corpus_entries))

    @property
    def keys(self):
        return [corpus_entry.id for corpus_entry in self.corpus_entries]

    @abstractmethod
    def train_dev_test_split(self):
        """return training-, validation- and test-set
        Since these sets are constructed
        """
        pass

    def summary(self):
        total_segments = [seg for entry in self.corpus_entries for seg in entry.segments]
        speeches = [seg for entry in self.corpus_entries for seg in entry.speech_segments]
        unaligned_speeches = [seg for entry in self.corpus_entries for seg in entry.speech_segments_unaligned]
        pauses = [seg for entry in self.corpus_entries for seg in entry.pause_segments]
        print('')
        print(f'Corpus: {self.name}')
        print('-----------------------------------------------------------')
        print(f'# corpus entries: {len(self.corpus_entries)}')
        print(f'# total segments: {len(total_segments)}')
        print(f'# speech segments: {len(speeches)}')
        print(f'# unaligned speech segments: {len(unaligned_speeches)}')
        print(f'# pause segments: {len(pauses)}')


class ReadyLinguaCorpus(Corpus):

    def __init__(self, corpus_entries, root_path):
        super().__init__('ReadyLingua', corpus_entries, root_path)

    def train_dev_test_split(self):
        n_entries = len(self.corpus_entries)
        # 80/10/10 split
        train_split = int(n_entries * 0.8)
        test_split = int(train_split + (n_entries - train_split) / 2)

        train_set = self.corpus_entries[:train_split]
        dev_set = self.corpus_entries[train_split:test_split]
        test_set = self.corpus_entries[test_split:]
        return train_set, dev_set, test_set


class LibriSpeechCorpus(Corpus):

    def __init__(self, corpus_entries, root_path):
        super().__init__('LibriSpeech', corpus_entries, root_path)

    def train_dev_test_split(self):
        train_set = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'train-')
        dev_set = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'dev-')
        test_set = filter_corpus_entry_by_subset_prefix(self.corpus_entries, ['test-', 'unknown'])
        return train_set, dev_set, test_set


class CorpusEntry(object):
    # cache values
    _audio = None
    _rate = None

    def __init__(self, audio_file, segments, original_path='', parms={}):
        self.corpus = None
        self.audio_file = audio_file
        self.x_path = None  # will be set in create_labelled_data.py
        self.y_path = None  # will be set in create_labelled_data.py

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


class Segment(ABC):
    def __init__(self, start_frame, end_frame, transcription, alignment_type):
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.text = ''
        self._transcription = ''
        self.transcription = transcription.strip() if transcription else ''

        self.segment_type = alignment_type
        self.corpus_entry = None

    @property
    def audio(self):
        rate, audio = self.corpus_entry.audio
        return rate, audio[self.start_frame:self.end_frame]

    @property
    def transcription(self):
        return self._transcription

    @transcription.setter
    def transcription(self, transcription):
        self._transcription = transcription
        self.text = normalize(transcription)


class Speech(Segment):
    def __init__(self, start_frame, end_frame, transcription=''):
        super().__init__(start_frame, end_frame, transcription, 'speech')


class Pause(Segment):
    def __init__(self, start_frame, end_frame):
        super().__init__(start_frame, end_frame, '', 'pause')


class UnalignedSpeech(Segment):
    """special class for speech segments where the text is derived from the original book text but the exact start
    and end position of the speech in the audio signal is not known (segment may contain pauses at start, end or
    anywhere inside the audio signal that were not aligned)"""

    def __init__(self, start_frame, end_frame, transcription=''):
        super().__init__(start_frame, end_frame, transcription, 'speech*')
