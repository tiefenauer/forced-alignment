import os
from os import makedirs

from os.path import exists

from util.audio_util import read_audio, write_wav_file
from util.vad_util import extract_voice

in_file = r'../assets/demo_files/address.wav'
target_dir = os.path.join('..', 'tmp')
if not exists(target_dir):
    makedirs(target_dir)

if __name__ == '__main__':
    audio, rate = read_audio(in_file, sample_rate=16000, mono=True)

    voice_segments = extract_voice(audio, rate)

    for i, voice in enumerate(voice_segments):
        out_file = os.path.join(target_dir, f'chunk-{i:0002d}.wav')
        print(f'Writing {out_file}')
        write_wav_file(out_file, voice.audio, voice.rate)
