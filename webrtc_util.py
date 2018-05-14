import collections

from webrtcvad import Vad


class Frame(object):
    def __init__(self, audio, timestamp, duration):
        self.audio = audio
        self.timestamp = timestamp
        self.duration = duration


def calculate_boundaries(corpus_entry, vad=Vad(3)):
    padding_duration_ms = 30
    frame_duration_ms = 30

    voiced_segments = []
    unvoiced_segments = []
    rate, audio = corpus_entry.audio

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    unvoiced_frames = []

    # frames = [audio[i:i + frame_duration_ms] for i in range(0, len(audio), frame_duration_ms)]
    # print(f'len(frames): {len(frames)}')
    frames = generate_frames(audio, rate)
    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.audio, rate)
        ring_buffer.append((frame, is_speech))

        if not triggered:
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames += [frame for frame, _, in ring_buffer]
                ring_buffer.clear()
            else:
                unvoiced_frames.append(frame)
        else:
            if unvoiced_frames:
                unvoiced_segments.append(unvoiced_frames)
                unvoiced_frames = []

            voiced_frames.append(frame)
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                voiced_segments.append(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []

    if voiced_frames:
        voiced_segments.append(voiced_frames)
    if unvoiced_frames:
        unvoiced_segments.append(unvoiced_frames)

    speech_boundaries = []
    for frames in voiced_segments:
        start = frames[0].timestamp * rate
        end = (frames[-1].timestamp + frames[-1].duration) * rate
        speech_boundaries.append((start, end))

    pause_boundaries = []
    for frames in unvoiced_segments:
        start = frames[0].timestamp * rate
        end = (frames[-1].timestamp + frames[-1].duration) * rate
        pause_boundaries.append((start, end))

    return speech_boundaries, pause_boundaries


def split_into_segments(corpus_entry, vad=Vad(3)):
    rate, audio = corpus_entry.audio
    frames = generate_frames(audio, rate)
    frames = list(frames)
    print(f'frames: {len(frames)}')
    voiced_segments = create_voiced_segments(rate, frames, vad)
    voiced_segments = list(voiced_segments)

    unvoiced_segments = create_unvoiced_segments(audio, voiced_segments)
    print(f'voiced_segments: {len(voiced_segments)}')
    print(f'unvoiced_segments: {len(unvoiced_segments)}')
    return voiced_segments


def generate_frames(audio, rate, frame_duration_ms=30):
    n_frames = int(rate * (frame_duration_ms / 1000.0)) * 2
    offset = 0
    timestamp = 0.0
    duration = (float(n_frames) / rate) / 2.0
    while offset + n_frames < len(audio):
        yield Frame(audio[offset:offset + n_frames], timestamp, duration)
        timestamp += duration
        offset += n_frames


def create_voiced_segments(sample_rate, audio, vad, frame_duration_ms=30, padding_duration_ms=30):
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


def create_unvoiced_segments(audio, voiced_segments):
    return []
