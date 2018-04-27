import audioop
import logging
import os
import wave

import numpy as np
import scipy.io.wavfile
import scipy.signal
from pydub import AudioSegment
from pydub.utils import mediainfo

log = logging.getLogger(__name__)


def resample_wav(src, dst, inrate=44100, outrate=16000, inchannels=1, outchannels=1, overwrite=False):
    """ Downsample WAV file to 16kHz
    Source: https://github.com/rpinsler/deep-speechgen/blob/master/downsample.py
    """

    # Skip if target file already exists
    if os.path.exists(dst) and not overwrite:
        return dst

    try:
        os.remove(dst)
    except OSError:
        pass

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    with wave.open(src, 'r') as s_read:
        try:
            n_frames = s_read.getnframes()
            data = s_read.readframes(n_frames)
            converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
            converted = converted[0]
            if outchannels == 1 and inchannels != 1:
                converted = audioop.tomono(converted, 2, 1, 0)
        except BaseException as e:
            log.error(f'Could not resample audio file {src}: {e}')

    with wave.open(dst, 'w') as s_write:
        try:
            s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
            s_write.writeframes(converted)
        except BaseException as e:
            log.error(f'Could not write resampled data {dst}: {e}')

    return dst


def read_wav_file(file_path):
    return scipy.io.wavfile.read(file_path)


def recalculate_frame(old_frame, old_sampling_rate=44100, new_sampling_rate=16000):
    factor = new_sampling_rate / old_sampling_rate
    new_frame = int(old_frame * factor)
    return new_frame


def calculate_frame(time_in_seconds, sampling_rate=16000):
    time_in_seconds = float(time_in_seconds)
    frame = int(time_in_seconds * sampling_rate)
    return frame


def mp3_to_wav(infile, outfile, outrate=16000, outchannels=1, overwrite=False):
    # Skip if target file already exists
    if os.path.exists(outfile) and not overwrite:
        return outfile

    info = mediainfo(infile)
    inrate = int(float(info['sample_rate']))
    inchannels = int(info['channels'])
    AudioSegment.from_mp3(infile).export(outfile, format="wav", parameters="-sample_rate 16000")
    if inrate != outrate:
        outfile_resampled = outfile + '.resampled'
        resample_wav(outfile, outfile_resampled, inrate, outrate, inchannels, outchannels)
        os.remove(outfile)
        os.rename(outfile_resampled, outfile)
    return outfile


def calculate_spectrogram(wav_file, nfft=200, fs=8000, noverlap=120):
    """
    Calculates the spectrogram of a WAV file
    :param wav_file: absolute path to the WAV file
    :param nfft: Length of each window segment
    :param fs: Sampling frequencies
    :param noverlap: Overlap between windows
    :return: (freqs, t, spec) the frequencies, times and spectrogram
    """
    rate, data = read_wav_file(wav_file)

    # only use one channel if WAV file is stereo
    x = data[:, 0] if data.ndim > 1 else data

    # calculate spectrogram
    freqs, t, spec = scipy.signal.spectrogram(x, nperseg=nfft, fs=fs, noverlap=noverlap)

    # rescale values to dB scale
    spec = 10. * np.log10(spec)
    spec = np.flipud(spec)

    return freqs, t, spec
