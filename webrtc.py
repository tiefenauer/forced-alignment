import contextlib
import wave

import webrtc_util
from corpus_util import load_corpus


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def main():
    rl_corpus = load_corpus(r'E:\readylingua-corpus\readylingua.corpus')
    # corpus_entry = random.choice(rl_corpus)
    corpus_entry = rl_corpus[0]

    sample_rate, audio = corpus_entry.audio
    voiced_segments, unvoiced_segments = webrtc_util.split_into_segments(audio, sample_rate)
    for i, voiced_segment in enumerate(voiced_segments):
        audio = b''.join([frame.audio for frame in voiced_segment])
        path = 'chunk-%002d.wav' % (i,)
        print(' Writing %s' % (path,))
        # write_wav_file(path, sample_rate, segment)
        write_wave(path, audio, sample_rate)


if __name__ == '__main__':
    main()
