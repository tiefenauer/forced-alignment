from tqdm import tqdm

from smith_waterman import smith_waterman
from util.vad_util import VoiceActivity


class Alignment(VoiceActivity):

    def __init__(self, va, text_start, text_end, alignment_text):
        super().__init__(va.audio, va.rate, va.start_frame, va.end_frame)
        self.text_start = text_start
        self.text_end = text_end
        self.alignment_text = alignment_text.strip()


def align(voice_activities, transcript, printout=False):
    alignments = []
    for va in tqdm(voice_activities, unit='voice activities'):
        b = va.transcript
        a = transcript  # transcript[end:]  # transcript[:len(transcript)//2]
        start, end, b_ = smith_waterman(a, b)
        alignment_text = transcript[start:end]
        if printout:
            print(f'transcript: {va.transcript} alignment: {alignment_text}')
        alignments.append(Alignment(va, start, end, alignment_text))

        # clip first part of transcript that was matched
        clip = transcript.index(alignment_text) + len(alignment_text)
        transcript = transcript[clip:]
    return alignments
