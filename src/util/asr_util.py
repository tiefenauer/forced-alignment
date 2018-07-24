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

from constants import LANGUAGE_CODES


def transcribe(voice_segments, language, printout=False):
    progress = tqdm(voice_segments, unit='voice activities')
    for voice in progress:
        transcript = transcribe_audio(voice.audio, voice.rate, language)
        progress.set_description(transcript)
        if printout and type(printout) is bool:
            print(transcript)
        voice.transcript = transcript

    if printout and type(printout) is str:
        print(f'saving ASR-transcripts to {printout}')
        with open(printout, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(voice.transcript for voice in voice_segments))
    return voice_segments


def transcribe_audio(audio, rate, language):
    # ugly hack to create temporary file until a solution is found to call the TTS-API directly
    tmp_file = 'audio.wav.tmp'
    sf.write(tmp_file, audio, rate, format='wav', subtype='PCM_16')
    transcription = transcribe_file(tmp_file, language)
    os.remove(tmp_file)
    return transcription


def transcribe_file(path, language):
    # transcribes a file --> should be changed to transcribe audio-bytes directly
    client = speech.SpeechClient()
    with io.open(path, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    language_code = LANGUAGE_CODES[language]
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code)

    # Detects speech in the audio file
    response = client.recognize(config, audio)

    if response and response.results:
        return response.results[0].alternatives[0].transcript
    return ''
