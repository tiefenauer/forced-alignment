import audioop
import logging
import os
import wave

import numpy as np
import scipy.io.wavfile
import scipy.signal
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
    wav_rate, wav_data = read_wav_file(wav_file)
    crop_start = min(segment.start_frame for segment in segments)
    crop_end = max(segment.end_frame for segment in segments)
    write_wav_file(wav_file, wav_rate, wav_data[crop_start:crop_end])

    for segment in segments:
        segment.start_frame -= crop_start
        segment.end_frame -= crop_start


def read_wav_file(file_path):
    return scipy.io.wavfile.read(file_path)


def write_wav_file(file_path, wav_rate, wav_data):
    scipy.io.wavfile.write(file_path, wav_rate, wav_data)


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


def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    # https://www.kaggle.com/davids1992/speech-representation-and-data-exploration

    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = scipy.signal.spectrogram(audio,
                                                  fs=sample_rate,
                                                  window='hann',
                                                  nperseg=nperseg,
                                                  noverlap=noverlap,
                                                  detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)
