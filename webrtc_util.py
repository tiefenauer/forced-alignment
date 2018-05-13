import collections

from webrtcvad import Vad


class Frame(object):
    def __init__(self, audio, timestamp, duration):
        self.audio = audio
        self.timestamp = timestamp
        self.duration = duration


def split_into_segments(corpus_entry, vad=Vad(3)):
    rate, audio = corpus_entry.audio
    frames = generate_frames(audio, rate)
    frames = list(frames)
    print(f'frames: {len(frames)}')
    segments = create_segments(rate, vad, frames)
    segments = list(segments)
    print(f'segments: {len(segments)}')
    return segments


def generate_frames(audio, rate, frame_duration_ms=30):
    n_frames = int(rate * (frame_duration_ms / 1000.0)) * 2
    offset = 0
    timestamp = 0.0
    duration = (float(n_frames) / rate) / 2.0
    while offset + n_frames < len(audio):
        yield Frame(audio[offset:offset + n_frames], timestamp, duration)
        timestamp += duration
        offset += n_frames


def create_segments(sample_rate, vad, audio, frame_duration_ms=30, padding_duration_ms=30):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in audio:
        is_speech = vad.is_speech(frame.audio, sample_rate)
        ring_buffer.append((frame, is_speech))

        if not triggered:
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames += [frame for frame, _, in ring_buffer]
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.audio for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []

    # leftover frames
    if voiced_frames:
        yield b''.join([f.audio for f in voiced_frames])
