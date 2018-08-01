from os import makedirs
from os.path import exists, join

from librosa.output import write_wav

from util.audio_util import read_audio
from util.vad_util import extract_voice

# in_file = r'D:\code\ip8\assets\demo_files\andiefreude.wav'
in_file = r'D:\code\ip8\assets\demo_files\address.mp3'
target_dir = join('..', 'tmp')
if not exists(target_dir):
    makedirs(target_dir)

if __name__ == '__main__':
    audio, rate = read_audio(in_file, resample_rate=16000, to_mono=True)
    voice_segments = extract_voice(audio, rate)

    for i, voice in enumerate(voice_segments):
        out_file = join(target_dir, f'chunk-{i:0002d}.wav')
        print(f'Writing {out_file}')
        write_wav(out_file, voice.audio, voice.rate)
