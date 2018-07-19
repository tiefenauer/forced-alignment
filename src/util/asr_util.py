"""
Utilities for ASR tasks
IMPORTANT: If the Google Cloud Speech API is used, you need to store your credentials as JSON and set an environment variable GOOGLE_APPLICATION_CREDENTIALS pointing to the location of the file!
See: https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries
"""
import io
import os

import soundfile as sf
# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from tqdm import tqdm

from util.webrtc_util import extract_speech


def transcribe_corpus_entry(audio, rate, limit=None):
    transcripts = []
    speech_audio = extract_speech(audio, rate)
    print(f"got {len(speech_audio)} segments, transcribing {limit if limit else 'all'}")
    for audio in tqdm(speech_audio[:limit], unit='segments'):
        transcript = transcribe_audio(audio, rate)
        transcripts.append(transcript)
    return transcripts


def transcribe_audio(audio, rate):
    tmp_file = 'audio.wav.tmp'
    sf.write(tmp_file, audio, rate, format='wav', subtype='PCM_16')
    transcription = transcribe_file(tmp_file)
    os.remove(tmp_file)
    return transcription


def transcribe_file(path):
    client = speech.SpeechClient()
    with io.open(path, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US')

    # Detects speech in the audio file
    response = client.recognize(config, audio)

    if response and response.results:
        return response.results[0].alternatives[0].transcript
    return ''
