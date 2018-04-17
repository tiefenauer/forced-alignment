import audioop
import logging
import os
import sys
import wave

import scipy.io.wavfile

log = logging.getLogger(__name__)


def resample_wav(src, dst, inrate=44100, outrate=16000, inchannels=1, outchannels=1):
    """ Downsample WAV file to 16kHz
    Source: https://github.com/rpinsler/deep-speechgen/blob/master/downsample.py
    """

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
    rate, data = scipy.io.wavfile.read(file_path)
    return data


def calculate_frame(old_frame, sampling_rate_old=44100, sampling_rate_new=16000):
    factor = sampling_rate_new / sampling_rate_old
    new_frame = int(old_frame * factor)
    return new_frame
