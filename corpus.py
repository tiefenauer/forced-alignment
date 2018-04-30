from abc import ABC, abstractmethod

from audio_util import read_wav_file
from corpus_util import filter_corpus_entry_by_subset_prefix


class Corpus(ABC):

    def __init__(self, name, corpus_entries):
        self.name = name
        self.corpus_entries = corpus_entries

    def __iter__(self):
        for corpus_entry in self.corpus_entries:
            yield corpus_entry

    def __getitem__(self, item):
        return self.corpus_entries[item]

    def __len__(self):
        return len(self.corpus_entries)

    @abstractmethod
    def train_dev_test_split(self):
        """return training-, validation- and test-set
        Since these sets are constructed
        """
        pass


class Alignment(ABC):
    def __init__(self, start_frame, end_frame, start_text, end_text, alignment_type):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_text = start_text
        self.end_text = end_text
        self.alignment_type = alignment_type
        self.corpus_entry = None

    @property
    def audio(self):
        _, audio = read_wav_file(self.corpus_entry.audio_file)
        return audio[self.start_frame:self.end_frame]

    @property
    def text(self):
        if self.start_text and self.end_text:
            return self.corpus_entry.transcript[self.start_text: self.end_text]
        return ''


class ReadyLinguaCorpus(Corpus):

    def __init__(self, corpus_entries):
        super().__init__('ReadyLingua', corpus_entries)

    def train_dev_test_split(self):
        # TODO: Implement split
        return [], [], []


class LibriSpeechCorpus(Corpus):

    def __init__(self, corpus_entries):
        super().__init__('LibriSpeech', corpus_entries)

    def train_dev_test_split(self):
        train_set = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'train-')
        dev_set = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'dev-')
        test_set = filter_corpus_entry_by_subset_prefix(self.corpus_entries, ['test-', 'unknown'])
        return train_set, dev_set, test_set


class CorpusEntry(object):
    def __init__(self, audio_file, transcript, alignments, original_path='', parms={}):
        self.audio_file = audio_file
        self.transcript = transcript
        for alignment in alignments:
            alignment.corpus_entry = self
        self.alignments = alignments

        self.original_path = original_path
        self.name = parms['name'] if 'name' in parms else ''
        self.language = parms['language'] if 'language' in parms else 'unknown'
        self.chapter_id = parms['chapter_id'] if 'chapter_id' in parms else 'unknown'
        self.speaker_id = parms['speaker_id'] if 'speaker_id' in parms else 'unkown'
        self.original_sampling_rate = parms['rate'] if 'rate' in parms else 'unknown'
        self.original_channels = parms['channels'] if 'channels' in parms else 'unknown'
        self.subset = parms['subset'] if 'subset' in parms else 'unknown'
        self.media_info = parms['media_info'] if 'media_info' in parms else {}

    def __iter__(self):
        for alignment in self.speech_segments:
            yield alignment

    def __getitem__(self, item):
        return self.speech_segments[item]

    @property
    def speech_segments(self):
        return [alignment for alignment in self.alignments if alignment.alignment_type == 'speech']

    @property
    def pause_segments(self):
        return [alignment for alignment in self.alignments if alignment.alignment_type == 'pause']


class Speech(Alignment):
    def __init__(self, start_frame, end_frame, start_text, end_text):
        super().__init__(start_frame, end_frame, start_text, end_text, 'speech')


class Pause(Alignment):
    def __init__(self, start_frame, end_frame):
        super().__init__(start_frame, end_frame, None, None, 'pause')
