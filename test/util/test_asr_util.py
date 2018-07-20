import unittest

import soundfile as sf

from util.asr_util import transcribe_audio, transcribe_file
from util.audio_util import write_wav_file, read_wav_file
from util.corpus_util import load_corpus


class TestAsrUtil(unittest.TestCase):

    @unittest.skip('use this to create samples')
    def test_create_samples_en(self):
        corpus = load_corpus(r'E:\readylingua-corpus')
        corpus = corpus(languages='en')
        corpus_entry = corpus[3]
        speech_segment = corpus_entry.speech_segments_not_numeric[0]
        write_wav_file('corpus_entry_sample_en.wav', corpus_entry.audio, corpus_entry.rate)
        write_wav_file('speech_segment_sample_en.wav', speech_segment.audio, speech_segment.rate)
        sf.write('speech_segment_sample_en_16.wav', speech_segment.audio, speech_segment.rate, subtype='PCM_16')
        sf.write('speech_segment_sample_en_16.flac', speech_segment.audio, speech_segment.rate, format='flac')
        sf.write('speech_segment_sample_en.flac', speech_segment.audio, speech_segment.rate, format='flac',
                 subtype='PCM_24')

    def test_transcribe_file(self):
        path = 'speech_segment_sample_en_16.wav'
        transcript = transcribe_file(path, 'en')
        print(transcript)

    def test_transcribe_audio_from_file(self):
        audio, rate = read_wav_file('speech_segment_sample_en.wav')
        transcription = transcribe_audio(audio, rate, 'en')
        print(transcription)

    def test_transcribe_audio(self):
        ls_corpus = load_corpus(r'E:\librispeech-corpus')
        corpus_entry = ls_corpus[0]
        speech_segment = corpus_entry.speech_segments_not_numeric[0]
        transcription = transcribe_audio(speech_segment.audio, speech_segment.rate, 'en')
        print(transcription)
