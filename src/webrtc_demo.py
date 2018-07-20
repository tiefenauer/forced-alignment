import os
from os import makedirs

import numpy as np
from os.path import exists

from util import vad_util
from util.audio_util import read_wav_file, write_wav_file, resample_wav

in_file = r'D:\corpus\readylingua-raw\Englisch\News in Englisch\2018\News180405\News180405.wav'
target_dir = os.path.join('..', 'tmp')
if not exists(target_dir):
    makedirs(target_dir)

if __name__ == '__main__':
    in_file_resampled = os.path.join(target_dir, 'resmpled.wav')
    _, rate = read_wav_file(in_file)
    resample_wav(in_file=in_file, out_file=in_file_resampled, inrate=rate)
    audio, rate = read_wav_file(in_file_resampled)

    voiced_segments, unvoiced_segments = vad_util.split_segments(audio, rate)

    for i, voiced_segment in enumerate(voiced_segments):
        audio = np.concatenate([frame.audio for frame in voiced_segment])
        out_file = os.path.join(target_dir, f'chunk-{i:0002d}.wav')
        print(f'Writing {out_file}')
        write_wav_file(out_file, audio, rate)
