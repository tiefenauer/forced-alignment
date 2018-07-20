import os
from os import makedirs

import numpy as np
from os.path import exists

from util import vad_util
from util.audio_util import read_audio, write_wav_file

in_file = r'../assets/demo_files/address.wav'
target_dir = os.path.join('..', 'tmp')
if not exists(target_dir):
    makedirs(target_dir)

if __name__ == '__main__':
    audio, rate = read_audio(in_file, sample_rate=16000, mono=True)

    voiced_segments, unvoiced_segments = vad_util.split_segments(audio, rate, aggressiveness=3)

    for i, voice in enumerate(voiced_segments):
        audio = np.concatenate([frame.audio for frame in voice])
        out_file = os.path.join(target_dir, f'chunk-{i:0002d}.wav')
        print(f'Writing {out_file}')
        write_wav_file(out_file, audio, rate)
