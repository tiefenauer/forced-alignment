import collections
import librosa
from webrtcvad import Vad

from util.audio_util import ms_to_frames


class Voice(object):
    """
    class representing voice activity inside an audio signal
    """

    def __init__(self, audio, rate, start_frame, end_frame):
        self._audio = audio
        self.rate = rate
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.transcript = None

    @property
    def audio(self):
        return self._audio[self.start_frame:self.end_frame]


class Frame(object):
    """
    object holding the audio signal of a fixed time interval (30ms) inside a long audio signal
    """

    def __init__(self, audio, timestamp, duration):
        self.audio = audio
        self.timestamp = timestamp
        self.duration = duration


def extract_voice(audio, rate, min_segments=2, max_segments=None):
    """
    Extract voice from audio usingn WebRTC (if possible) or by splitting into non-silent intervals (fallback)
    :param audio: audio signal as 1D-numpy array (mono)
    :param rate: sample rate
    :param min_segments: expected minimal number of segments
    :param max_segments: maximum nuber of segments to return
    :return:
    """
    voiced_segments = list(webrtc_voice(audio, rate, max_segments))
    if len(voiced_segments) >= min_segments:
        return voiced_segments
    return list(librosa_voice(audio, rate, limit=max_segments))


def webrtc_voice(audio, rate, limit=None):
    voiced_frames, _ = webrtc_split(audio, rate)
    for frames in voiced_frames[:limit]:
        start_time = frames[0].timestamp
        end_time = (frames[-1].timestamp + frames[-1].duration)
        start_frame = ms_to_frames(start_time * 1000, rate)
        end_frame = ms_to_frames(end_time * 1000, rate)
        yield Voice(audio, rate, start_frame, end_frame)


def librosa_voice(audio, rate, top_db=30, limit=None):
    intervals = librosa.effects.split(audio, top_db=top_db)
    for start, end in intervals[:limit]:
        yield Voice(audio, rate, start, end)


def webrtc_split(audio, rate, aggressiveness=3, window_duration_ms=30, frame_duration_ms=30):
    """
    Splic audio into voiced and unvoiced segments using WebRTC
    :param audio:
    :param rate:
    :param aggressiveness:
    :param window_duration_ms:
    :param frame_duration_ms:
    :return:
    """
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
