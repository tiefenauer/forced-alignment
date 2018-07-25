from os import remove

import collections
import librosa
import soundfile as sf
from webrtcvad import Vad

from util.audio_util import ms_to_frames, write_pcm16_wave, read_audio, write_wav_file, read_pcm16_wave


class Voice(object):
    """
    class representing voice activity inside an audio signal
    """

    def __init__(self, audio, rate, start_frame, end_frame):
        self.audio = audio
        self.rate = rate
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.transcript = None


def extract_voice(audio, rate, min_segments=2, max_segments=None):
    """
    Extract voice from audio usingn WebRTC (if possible) or by splitting into non-silent intervals (fallback)
    :param audio: audio signal as 1D-numpy array (mono)
    :param rate: sample rate
    :param min_segments: expected minimal number of segments
    :param max_segments: maximum nuber of segments to return
    :return:
    """
    voice_segments = list(webrtc_voice(audio, rate))
    if len(voice_segments) >= min_segments:
        return voice_segments
    return list(librosa_voice(audio, rate, limit=max_segments))


def webrtc_voice(audio, rate, aggressiveness=3):
    voiced_frames = webrtc_split(audio, rate, aggressiveness=aggressiveness)
    for voice_frames, voice_rate in voiced_frames:
        voice_bytes = b''.join([f.bytes for f in voice_frames])
        voice_audio, rate = from_pcm16(voice_bytes, rate)

        start_time = voice_frames[0].timestamp
        end_time = (voice_frames[-1].timestamp + voice_frames[-1].duration)
        start_frame = ms_to_frames(start_time * 1000, rate)
        end_frame = ms_to_frames(end_time * 1000, rate)
        yield Voice(voice_audio, voice_rate, start_frame, end_frame)


def librosa_voice(audio, rate, top_db=30, limit=None):
    intervals = librosa.effects.split(audio, top_db=top_db)
    for start, end in intervals[:limit]:
        yield Voice(audio, rate, start, end)


def webrtc_split(audio, rate, aggressiveness=3, frame_duration_ms=30, window_duration_ms=300):
    # adapted from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    audio_bytes, audio_rate = to_pcm16(audio, rate)

    vad = Vad(aggressiveness)
    num_window_frames = int(window_duration_ms / frame_duration_ms)
    sliding_window = collections.deque(maxlen=num_window_frames)
    triggered = False

    voiced_frames = []
    for frame in generate_frames(audio_bytes, audio_rate, frame_duration_ms):
        is_speech = vad.is_speech(frame.bytes, audio_rate)
        sliding_window.append((frame, is_speech))

        if not triggered:
            num_voiced = len([f for f, speech in sliding_window if speech])
            if num_voiced > 0.9 * sliding_window.maxlen:
                triggered = True
                voiced_frames += [frame for frame, _ in sliding_window]
                sliding_window.clear()
        else:
            voiced_frames.append(frame)
            num_unvoiced = len([f for f, speech in sliding_window if not speech])
            if num_unvoiced > 0.9 * sliding_window.maxlen:
                triggered = False
                yield voiced_frames, audio_rate
                sliding_window.clear()
                voiced_frames = []
    if voiced_frames:
        yield voiced_frames, audio_rate


class Frame(object):
    """
    object holding the audio signal of a fixed time interval (30ms) inside a long audio signal
    """

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def generate_frames(audio, sample_rate, frame_duration_ms=30):
    frame_length = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    timestamp = 0.0
    duration = (float(frame_length) / sample_rate) / 2.0
    while offset + frame_length < len(audio):
        yield Frame(audio[offset:offset + frame_length], timestamp, duration)
        timestamp += duration
        offset += frame_length


def to_pcm16(audio, rate):
    """
    convert any audio signal to PCM_16 that can be understood by WebRTC-VAD
    :param audio: audio signal (arbitrary source format, number of channels or encoding)
    :param rate: sampling rate (arbitrary value)
    :return: a PCM16-Encoded Byte array of the signal converted to 16kHz (mono)
    """
    tmp_file = 'tmp.wav'
    # convert to 16kHz (mono) if neccessary
    if rate != 16000 or audio.ndim > 1:
        write_wav_file(tmp_file, audio, rate)
        audio, rate = read_audio(tmp_file, sample_rate=16000, mono=True)
    # convert to PCM_16
    sf.write(tmp_file, audio, rate, subtype='PCM_16')
    audio, rate = read_pcm16_wave(tmp_file)
    return audio, rate


def from_pcm16(bytes, rate):
    """
    convert PCM_16 audio to 32-Bit float LE (f32l)
    :param bytes: PCM_16 encoded audio bytes
    :param rate:
    :return:
    """
    tmp_file = 'tmp.wav'
    write_pcm16_wave(tmp_file, bytes, rate)
    audio, rate = read_audio(tmp_file)
    remove(tmp_file)
    return audio, rate
