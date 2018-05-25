from abc import ABC

from util.string_util import normalize


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