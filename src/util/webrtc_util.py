import collections
import numpy as np
from webrtcvad import Vad


class Frame(object):
    def __init__(self, audio, timestamp, duration):
        self.audio = audio
        self.timestamp = timestamp
        self.duration = duration


def extract_speech(corpus_entry):
    voiced_segments, _ = split_segments(corpus_entry)
    speech_audio = []
    for frames in voiced_segments:
        speech_audio.append(np.concatenate([frame.audio for frame in frames]))
    return speech_audio


def split_segments(corpus_entry, aggressiveness=3, window_duration_ms=30, frame_duration_ms=30):
    # https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    vad = Vad(aggressiveness)

    # create sliding window
    num_window_frames = int(window_duration_ms / frame_duration_ms)
    sliding_window = collections.deque(maxlen=num_window_frames)
    triggered = False

    # initialize
    voiced_segments, unvoiced_segments, voiced_frames, unvoiced_frames = [], [], [], []

    frames = generate_frames(corpus_entry.audio, corpus_entry.rate, frame_duration_ms)
    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.audio, corpus_entry.rate)
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
