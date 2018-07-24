import collections
import contextlib
import wave
from os import makedirs
from os.path import join, exists

import librosa
import webrtcvad


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


# in_file = r'D:\code\ip8\assets\demo_files\andiefreude_44.wav'
# in_file = r'D:\code\ip8\assets\demo_files\andiefreude_16.wav'
in_file = r'D:\code\ip8\assets\demo_files\address.wav'
# in_file = r'D:\code\ip8\assets\demo_files\address.mp3'
target_dir = join('..', 'tmp')
if not exists(target_dir):
    makedirs(target_dir)

tmp_file = join(target_dir, 'tmp.wav')


def main():
    import numpy as np
    # sample_rate, samples = wavfile.read(in_file)
    import wavio
    wavio_audio = wavio.read(in_file)

    audio, rate = librosa.load(in_file, sr=16000, mono=True, dtype=np.uint8)
    import struct
    audio = struct.pack("%dh" % len(audio), *audio)
    # audio, rate = read_audio(in_file, 16000, True)
    import soundfile as sf
    sf.write(tmp_file, audio, rate, subtype='PCM_16')
    sf.read()
    audio, rate = read_wave(tmp_file)

    vad = webrtcvad.Vad(3)
    frames = frame_generator(30, audio, rate)
    frames = list(frames)
    segments = vad_collector(rate, 30, 300, vad, frames)
    for i, segment in enumerate(segments):
        path = join(target_dir, 'chunk-%002d.wav' % (i,))
        print(' Writing %s' % (path,))
        write_wave(path, segment, rate)


if __name__ == '__main__':
    main()
