import audioop
import logging
import os
import random
import wave

import librosa
import numpy as np
import scipy.io.wavfile
import scipy.signal
from librosa.effects import time_stretch, pitch_shift
from pydub import AudioSegment

log = logging.getLogger(__name__)


def resample_wav(in_file, out_file, inrate=44100, outrate=16000, inchannels=1, outchannels=1):
    """ Downsample WAV file to 16kHz
    Source: https://github.com/rpinsler/deep-speechgen/blob/master/downsample.py
    """

    try:
        os.remove(out_file)
    except OSError:
        pass

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    with wave.open(in_file, 'r') as s_read:
        try:
            n_frames = s_read.getnframes()
            data = s_read.readframes(n_frames)
            converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
            converted = converted[0]
            if outchannels == 1 and inchannels != 1:
                converted = audioop.tomono(converted, 2, 1, 0)
        except BaseException as e:
            log.error(f'Could not resample audio file {in_file}: {e}')

    with wave.open(out_file, 'w') as s_write:
        try:
            s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
            s_write.writeframes(converted)
        except BaseException as e:
            log.error(f'Could not write resampled data {out_file}: {e}')


def crop_wav(wav_file, segments):
    audio, rate = read_wav_file(wav_file)
    crop_start = min(segment.start_frame for segment in segments)
    crop_end = max(segment.end_frame for segment in segments)
    write_wav_file(wav_file, audio[crop_start:crop_end], rate)

    for segment in segments:
        segment.start_frame -= crop_start
        segment.end_frame -= crop_start


def read_wav_file(file_path):
    return librosa.load(file_path, sr=None)


def write_wav_file(file_path, audio, rate):
    librosa.output.write_wav(file_path, audio, rate)


def recalculate_frame(old_frame, old_sampling_rate=44100, new_sampling_rate=16000):
    factor = new_sampling_rate / old_sampling_rate
    return int(old_frame * factor)


def calculate_frame(time_in_seconds, sampling_rate=16000):
    time_in_seconds = float(time_in_seconds)
    frame = int(time_in_seconds * sampling_rate)
    return frame


def mp3_to_wav(infile, outfile, outrate=16000, outchannels=1):
    AudioSegment.from_mp3(infile) \
        .set_frame_rate(outrate) \
        .set_channels(outchannels) \
        .export(outfile, format="wav")


def calculate_spectrogram(audio, sample_rate, nperseg=200, noverlap=120):
    """
    Calculates the spectrogram of a WAV file
    :param audio: numpy array containing the audio data
    :param sample_rate: sampling rate of the audio data
    :param nperseg: length of each window segment
    :param noverlap: overlap between windows
    :return: (freqs, times, spec) the frequencies (numpy 1D array), time steps (numpy 1-D array) and spectrogram (numpy 2D array)
    """

    # only use one channel if WAV file is stereo
    audio = audio[:, 0] if audio.ndim > 1 else audio

    # calculate spectrogram
    freqs, times, spec = scipy.signal.spectrogram(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    # rescale values to dB scale
    spec = 10. * np.log10(spec)
    spec = np.flipud(spec)

    return freqs, times, spec


# @deprecated(reason='spectrogram calculation with scipy has been replaced by librosa. Use log_spectgram instead')
def log_specgram(audio, sample_rate, window_size=20, step_size=10, unit='ms', mode='psd'):
    # https://www.kaggle.com/davids1992/speech-representation-and-data-exploration

    if unit == 'ms':
        window_size = ms_to_frames(window_size, sample_rate)
        step_size = ms_to_frames(step_size, sample_rate)

    freqs, times, spec = scipy.signal.spectrogram(audio, mode=mode,
                                                  fs=sample_rate,
                                                  window='hann',
                                                  nperseg=window_size,
                                                  noverlap=step_size,
                                                  detrend=False)

    return freqs, times, np.log(spec.astype(np.float32) + 1e-10)


def ms_to_frames(val_ms, sample_rate):
    return int(round(val_ms * sample_rate / 1e3))


def shift(audio, max_shift=None):
    max_shift = max_shift if max_shift else int(0.01 * len(audio))
    shift = np.random.randint(low=1, high=max_shift)
    return audio[shift:]


def distort(audio, rate, tempo=False, pitch=False):
    audio = audio.astype(np.float32)
    distorted = audio
    if tempo:
        factor = random.uniform(0.8, 1.2) if isinstance(tempo, bool) else tempo
        distorted = time_stretch(distorted, factor)
    if pitch:
        factor = random.uniform(1, 4) if isinstance(tempo, bool) else tempo
        distorted = pitch_shift(distorted, rate, factor)
    return distorted
