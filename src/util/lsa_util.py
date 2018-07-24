from pattern3.metrics import levenshtein_similarity
from tqdm import tqdm

from smith_waterman import smith_waterman
from util.vad_util import Voice


class Alignment(Voice):

    def __init__(self, va, text_start, text_end, alignment_text):
        super().__init__(va.audio, va.rate, va.start_frame, va.end_frame)
        self.text_start = text_start
        self.text_end = text_end
        self.alignment_text = alignment_text.strip()


def align(voice_activities, transcript, printout=False):
    a = transcript  # transcript[end:]  # transcript[:len(transcript)//2]
    alignments, lines = [], []

    progress = tqdm([va for va in voice_activities if len(va.transcript.strip()) > 0], unit='voice activities')
    for va in progress:
        b = va.transcript
        start, end, b_ = smith_waterman(a, b)
        alignment_text = transcript[start:end]
        progress.set_description(alignment_text)

        edit_distance = levenshtein_similarity(va.transcript, alignment_text)
        line = f'edit distance: {edit_distance:.2f}, transcript: {va.transcript}, alignment: {alignment_text}'
        if printout and type(printout) is bool:
            print(line)
        lines.append(line)

        if edit_distance > 0.5:
            alignments.append(Alignment(va, start, end, alignment_text))
            # prevent identical transcripts to become aligned with the same part of the audio
            a = transcript.replace(alignment_text, '-' * len(alignment_text), 1)

    if printout and type(printout) is str:
        with open(printout, 'w') as f:
            f.writelines(lines)
    return alignments
