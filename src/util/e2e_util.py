"""
Utility functions for end-to-end tasks
"""
import json
import os
import pickle
from os import makedirs, remove
from os.path import join, exists, splitext, basename, relpath
from pathlib import Path

import langdetect
from bs4 import BeautifulSoup
from librosa.output import write_wav
from pydub import AudioSegment

from constants import DEMO_ROOT
from util.asr_util import transcribe
from util.audio_util import frame_to_ms, read_audio
from util.lsa_util import align
from util.vad_util import extract_voice


def create_demo(audio_path, transcript_path, id=None, limit=None):
    file_name, file_ext = splitext(basename(audio_path))
    demo_id = id if id else file_name
    print(f'assigned demo id: {demo_id}.')

    print(f'Loading audio and transcript...')
    audio, rate = read_audio(audio_path, 16000, True)
    transcript = Path(transcript_path).read_text(encoding='utf-8')
    print(f'... audio and transcript loaded')

    language = langdetect.detect(transcript)
    print(f'detected language from transcript: {language}')
    return create_demo_files(demo_id, audio, rate, transcript, language, limit=limit)


def create_demo_from_corpus_entry(corpus_entry, limit=None):
    demo_id = corpus_entry.id
    audio, rate = corpus_entry.audio, corpus_entry.rate
    transcript, language = corpus_entry.full_transcript, corpus_entry.language
    return create_demo_files(demo_id, audio, rate, transcript, language, limit=limit)


def create_demo_files(demo_id, audio, rate, transcript, language, limit=None):
    print(f'creating demo with id={demo_id}')
    target_dir = join(DEMO_ROOT, demo_id)
    if not exists(target_dir):
        makedirs(target_dir)
    print(f'all assets will be saved in {target_dir}')

    asr_pickle = join(target_dir, 'asr.pkl')  # cached STT-responses to regenerate files faster
    audio_path = join(target_dir, 'audio.mp3')
    transcript_path = join(target_dir, 'transcript.txt')
    transcript_asr_path = join(target_dir, 'transcript_asr.txt')
    alignment_text_path = join(target_dir, 'alignment.txt')
    alignment_json_path = join(target_dir, 'alignment.json')

    print(f'saving audio in {audio_path}')
    tmp_wav = join(target_dir, 'audio.tmp.wav')
    write_wav(tmp_wav, audio, rate)
    AudioSegment.from_wav(tmp_wav).export(audio_path, format='mp3')
    remove(tmp_wav)

    print(f'saving transcript in {transcript_path}')
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    transcript = transcript.replace('\n', ' ')

    if exists(asr_pickle):
        print(f'VAD + ASR: loading cached results from pickle: {asr_pickle}')
        with open(asr_pickle, 'rb') as f:
            voice_segments = pickle.load(f)
        with open(transcript_asr_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join([voice.transcript for voice in voice_segments]))
    else:
        print(f'VAD: Splitting audio into speech segments')
        voice_segments = extract_voice(audio, rate, max_segments=limit)
        print(f'ASR: transcribing each segment')
        voice_segments = transcribe(voice_segments, language, printout=transcript_asr_path)
        print(f'saving results to cache: {asr_pickle}')
        with open(asr_pickle, 'wb') as f:
            pickle.dump(voice_segments, f)

    print(f'aligning audio with transcript')
    alignments = align(voice_segments, transcript, printout=alignment_text_path)

    print(f'saving alignment information to {alignment_json_path}')
    json_data = create_alignment_json(alignments)
    with open(alignment_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    update_index(demo_id)
    demo_path = create_demo_index(target_dir, demo_id, transcript)
    return create_url(demo_path, target_dir)


def create_alignment_json(alignments):
    words = []
    for al in alignments:
        start_ms = frame_to_ms(al.start_frame, al.rate)
        end_ms = frame_to_ms(al.end_frame, al.rate)
        words.append([al.text, start_ms, end_ms])

    json_data = {}
    json_data['words'] = words
    return json_data


def create_demo_index(target_dir, demo_id, transcript):
    template_path = join(DEMO_ROOT, '_template.html')
    soup = BeautifulSoup(open(template_path), 'html.parser')
    soup.title.string = demo_id
    soup.find(id='demo_title').string = f'Forced Alignment for {demo_id}'
    soup.find(id='target').string = transcript.replace('\n', ' ')

    demo_path = join(target_dir, 'index.html')
    with open(demo_path, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())

    return demo_path


def update_index(demo_id):
    index_path = join(DEMO_ROOT, 'index.html')
    soup = BeautifulSoup(open(index_path), 'html.parser')

    if not soup.find(id=demo_id):
        a = soup.new_tag('a', href=demo_id)
        a.string = demo_id
        li = soup.new_tag('li', id=demo_id)
        li.append(a)
        ul = soup.find(id='demo_list')
        ul.append(li)

        with open(index_path, 'w') as f:
            f.write(soup.prettify())


def create_url(demo_path, target_dir):
    return 'https://ip8.tiefenauer.info:8888/' + relpath(demo_path, Path(target_dir).parent).replace(os.sep, '/')