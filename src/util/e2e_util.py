import json
import pickle
from os import makedirs
from pathlib import Path

import langdetect
import librosa
from bs4 import BeautifulSoup
from os.path import join, exists, splitext, basename

from constants import DEMO_ROOT
from util.asr_util import transcribe
from util.audio_util import write_wav_file, frame_to_ms
from util.lsa_util import align
from util.vad_util import extract_voice


def create_demo(audio_path, transcript_path, limit=None):
    demo_id = splitext(basename(audio_path))[0]
    print(f'assigned demo id: {demo_id}. Loading audio and transcript...')
    audio, rate = librosa.core.load(audio_path, sr=16000, mono=True)
    transcript = Path(transcript_path).read_text(encoding='utf-8').replace('\n', ' ')
    print(f'... audio and transcript loaded')
    language = langdetect.detect(transcript)
    print(f'detected language from transcript: {language}')
    create_demo_files(demo_id, audio, rate, transcript, language, limit)


def create_demo_from_corpus_entry(corpus_entry, limit=None):
    demo_id = corpus_entry.id
    audio, rate = corpus_entry.audio, corpus_entry.rate
    transcript, language = corpus_entry.full_transcript, corpus_entry.language
    create_demo_files(demo_id, audio, rate, transcript, language, limit)


def create_demo_files(demo_id, audio, rate, transcript, language, limit=None):
    print(f'creating demo with id={demo_id}')
    target_dir = join(DEMO_ROOT, demo_id)
    if not exists(target_dir):
        makedirs(target_dir)
    print(f'all assets will be saved in {target_dir}')

    asr_pickle = join(target_dir, 'asr.pkl')  # cached STT-responses to regenerate files faster
    audio_path = join(target_dir, 'audio.wav')
    transcript_path = join(target_dir, 'transcript.txt')
    transcript_asr_path = join(target_dir, 'transcript_asr.txt')
    alignment_json_path = join(target_dir, 'alignment.json')

    print(f'saving audio in {audio_path}')
    write_wav_file(audio_path, audio, rate)
    print(f'saving transcript in {transcript_path}')
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)

    if exists(asr_pickle):
        print(f'VAD + ASR: loading cached results from pickle: {asr_pickle}')
        with open(asr_pickle, 'rb') as f:
            voice_activities = pickle.load(f)
    else:
        print(f'VAD: Splitting audio into speech segments')
        voice_activities = extract_voice(audio, rate, limit)
        print(f'ASR: transcribing each segment')
        voice_activities = transcribe(voice_activities, language, printout=True)
        print(f'saving results to cache: {asr_pickle}')
        with open(asr_pickle, 'wb') as f:
            pickle.dump(voice_activities, f)

    print(f'saving ASR-transcripts to {transcript_asr_path}')
    with open(transcript_asr_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(va.transcript for va in voice_activities))

    print(f'aligning audio with transcript')
    alignment = align(voice_activities, transcript, printout=True)

    print(f'saving alignment information to {alignment_json_path}')
    json_data = create_alignment_json(alignment)
    with open(alignment_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    create_demo_index(target_dir, demo_id, transcript)
    update_index(demo_id)


def create_alignment_json(alignments):
    words = []
    for al in alignments:
        start_ms = frame_to_ms(al.start_frame, al.rate) * 2
        end_ms = frame_to_ms(al.end_frame, al.rate) * 2
        words.append([al.alignment_text, start_ms, end_ms])

    json_data = {}
    json_data['words'] = words
    return json_data


def create_demo_index(target_dir, demo_id, transcript):
    template_path = join(DEMO_ROOT, '_template.html')
    soup = BeautifulSoup(open(template_path), 'html.parser')
    soup.title.string = demo_id
    soup.find(id='demo_title').string = f'Forced Alignment for {demo_id}'
    soup.find(id='target').string = transcript.replace('\n', ' ')

    demo_html = join(target_dir, 'index.html')
    with open(demo_html, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())

    return demo_html


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
