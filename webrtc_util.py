import collections

from webrtcvad import Vad


class Frame(object):
    def __init__(self, audio, timestamp, duration):
        self.audio = audio
        self.timestamp = timestamp
        self.duration = duration


def calculate_boundaries(corpus_entry, vad=Vad(3)):
    sample_rate, audio = corpus_entry.audio

    voiced_segments, unvoiced_segments = split_into_segments(audio, sample_rate, vad)

    speech_boundaries = calculate_boundaries_from_segments(voiced_segments, sample_rate)
    pause_boundaries = calculate_boundaries_from_segments(unvoiced_segments, sample_rate)

    return speech_boundaries, pause_boundaries


def calculate_boundaries_from_segments(voiced_segments, sample_rate):
    boundaries = []
    for frames in voiced_segments:
        start = frames[0].timestamp * sample_rate
        end = (frames[-1].timestamp + frames[-1].duration) * sample_rate
        boundaries.append((start, end))
    return boundaries


def split_into_segments(audio, sample_rate, vad=Vad(3), window_duration_ms=30, frame_duration_ms=30):
    # https://github.com/wiseman/py-webrtcvad/blob/master/example.py

    # create sliding window
    num_window_frames = int(window_duration_ms / frame_duration_ms)
    sliding_window = collections.deque(maxlen=num_window_frames)
    triggered = False

    # initialize
    voiced_segments = []
    unvoiced_segments = []
    voiced_frames = []
    unvoiced_frames = []

    frames = generate_frames(audio, sample_rate, frame_duration_ms)
    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.audio, sample_rate)
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
    n_frames = int(sample_rate * (frame_duration_ms / 1000.0)) * 2
    offset = 0
    timestamp = 0.0
    duration = (float(n_frames) / sample_rate) / 2.0
    while offset + n_frames < len(audio):
        yield Frame(audio[offset:offset + n_frames], timestamp, duration)
        timestamp += duration
        offset += n_frames