from corpus.audible import Audible
from util.string_util import normalize


class Segment(Audible):
    def __init__(self, start_frame, end_frame, transcript, alignment_type):
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.text = ''
        self._transcript = ''
        self.transcript = transcript.strip() if transcript else ''

        self.segment_type = alignment_type
        self.corpus_entry = None

    @property
    def audio(self):
        return self.corpus_entry.audio[self.start_frame:self.end_frame]

    @property
    def rate(self):
        return self.corpus_entry.rate

    @property
    def transcript(self):
        return self._transcript

    @transcript.setter
    def transcript(self, transcript):
        self._transcript = transcript
        self.text = normalize(transcript)

    @property
    def audio_length(self):
        sample_rate = int(float(self.corpus_entry.media_info['sample_rate']))
        return (self.end_frame - self.start_frame) / sample_rate


class Speech(Segment):
    def __init__(self, start_frame, end_frame, transcript=''):
        super().__init__(start_frame, end_frame, transcript, 'speech')


class Pause(Segment):
    def __init__(self, start_frame, end_frame):
        super().__init__(start_frame, end_frame, '', 'pause')


class UnalignedSpeech(Segment):
    """special class for speech segments where the text is derived from the original book text but the exact start
    and end position of the speech in the audio signal is not known (segment may contain pauses at start, end or
    anywhere inside the audio signal that were not aligned)"""

    def __init__(self, start_frame, end_frame, transcript=''):
        super().__init__(start_frame, end_frame, transcript, 'speech*')
