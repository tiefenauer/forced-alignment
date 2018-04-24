from audio_util import read_wav_file


class Corpus(object):

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


class CorpusEntry(object):
    def __init__(self, audio_file, transcript, alignments, speech_pauses, original_path='', parms={}):
        self.audio_file = audio_file
        self.transcript = transcript
        for alignment in alignments:
            alignment.corpus_entry = self
        self.alignments = alignments
        for speech_pause in speech_pauses:
            speech_pause.corpus_entry = self
        self.speech_pauses = speech_pauses

        self.original_path = original_path
        self.name = parms['name'] if 'name' in parms else ''
        self.language = parms['language'] if 'language' in parms else 'unknown'
        self.original_sampling_rate = parms['rate'] if 'rate' in parms else 'unknown'
        self.original_channels = parms['channels'] if 'channels' in parms else 'unknown'

    def __iter__(self):
        for alignment in self.alignments:
            yield alignment

    def __getitem__(self, item):
        return self.alignments[item]


class Alignment(object):
    def __init__(self, start_frame, end_frame, start_text, end_text):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_text = start_text
        self.end_text = end_text
        self.corpus_entry = None

    @property
    def audio(self):
        audio = read_wav_file(self.corpus_entry.audio_file)
        return audio[self.start_frame:self.end_frame]

    @property
    def text(self):
        return self.corpus_entry.transcript[self.start_text: self.end_text]  # komische Indizierung...


class Segment(object):
    def __init__(self, start_frame, end_frame, segment_type):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.segment_type = segment_type
        self.corpus_entry = None

    @property
    def audio(self):
        audio = read_wav_file(self.corpus_entry.audio_file)
        return audio[self.start_frame:self.end_frame]