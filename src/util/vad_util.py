import collections
import numpy as np
from webrtcvad import Vad

from util.audio_util import ms_to_frames


class VoiceActivity(object):
    """
    class representing voice activity inside an audio signal
    """

    def __init__(self, audio, rate, start_frame, end_frame):
        self.audio = audio
        self.rate = rate
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.transcript = None


class Frame(object):
    """
    object holding the audio signal of a fixed time interval (30ms) inside a long audio signal
    """

    def __init__(self, audio, timestamp, duration):
        self.audio = audio
        self.timestamp = timestamp
        self.duration = duration


def extract_voice_activities(audio, rate, limit=None):
    voiced_segments, _ = split_segments(audio, rate)
    voice_activities = []
    for frames in voiced_segments[:limit]:
        start_time = frames[0].timestamp
        end_time = (frames[-1].timestamp + frames[-1].duration)
        start_frame = ms_to_frames(start_time * 1000, rate)
        end_frame = ms_to_frames(end_time * 1000, rate)
        audio = np.concatenate([frame.audio for frame in frames])
        va = VoiceActivity(audio, rate, start_frame, end_frame)
        voice_activities.append(va)
    return voice_activities


def split_segments(audio, rate, aggressiveness=3, window_duration_ms=30, frame_duration_ms=30):
    # https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    vad = Vad(aggressiveness)

    # create sliding window
    num_window_frames = int(window_duration_ms / frame_duration_ms)
    sliding_window = collections.deque(maxlen=num_window_frames)
    triggered = False

    # initialize
    voiced_segments, unvoiced_segments, voiced_frames, unvoiced_frames = [], [], [], []

    frames = generate_frames(audio, rate, frame_duration_ms)
    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.audio, rate)
        sliding_window.append((frame, is_speech))

        if not triggered:
            num_voiced = len([f for f, speech in sliding_window if speech])
            if num_voiced > 0.9 * sliding_window.maxlen:
                triggered = True
                voiced_frames += [frame for frame, _ in sliding_window]
                sliding_window.clear()
            else:
                unvoiced_frames.append(frame)
        else:
            if unvoiced_frames:
                unvoiced_segments.append(unvoiced_frames)
                unvoiced_frames = []

            voiced_frames.append(frame)
            num_unvoiced = len([f for f, speech in sliding_window if not speech])
            if num_unvoiced > 0.9 * sliding_window.maxlen:
                triggered = False
                voiced_segments.append(voiced_frames)
                sliding_window.clear()
                voiced_frames = []

    if voiced_frames:
        voiced_segments.append(voiced_frames)
    if unvoiced_frames:
        unvoiced_segments.append(unvoiced_frames)

    return voiced_segments, unvoiced_segments


def generate_frames(audio, sample_rate, frame_duration_ms=30):
    frame_length = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    timestamp = 0.0
    duration = (float(frame_length) / sample_rate) / 2.0
    while offset + frame_length < len(audio):
        yield Frame(audio[offset:offset + frame_length], timestamp, duration)
        timestamp += duration
        offset += frame_length
