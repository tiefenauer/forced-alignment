from os import makedirs
from os.path import exists, join

from librosa.output import write_wav

from util.audio_util import read_audio
from util.vad_util import webrtc_split, webrtc_voice

# in_file = r'D:\code\ip8\assets\demo_files\andiefreude.mp3'
in_file = r'D:\code\ip8\assets\demo_files\address.mp3'
target_dir = join('..', 'tmp')
if not exists(target_dir):
    makedirs(target_dir)

if __name__ == '__main__':
    audio, rate = read_audio(in_file)

    voice_segments = list(webrtc_voice(audio, rate))
    for i, voice in enumerate(voice_segments):
        out_file = join(target_dir, f'chunk-{i:0002d}.wav')
        print(f'Writing {out_file}')
        write_wav(out_file, voice.audio, voice.rate)

    voice_segments = webrtc_split(audio, rate)
