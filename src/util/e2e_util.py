import json
import pickle
from os import makedirs

from bs4 import BeautifulSoup
from os.path import join, exists

from constants import DEMO_ROOT
from util.asr_util import transcribe
from util.audio_util import write_wav_file, frame_to_ms
from util.lsa_util import align
from util.vad_util import extract_voice_activities


def create_demo_from_corpus_entry(corpus_entry, limit=None):
    sample_id = corpus_entry.id
    audio, rate = corpus_entry.audio, corpus_entry.rate
    transcript, language = corpus_entry.full_transcript, corpus_entry.language
    create_demo(sample_id, audio, rate, transcript, language, limit)


def create_demo(sample_id, audio, rate, transcript, language, limit=None):
    print(f'creating demo with id={sample_id}')
    target_dir = join(DEMO_ROOT, sample_id)
    if not exists(target_dir):
        makedirs(target_dir)
    print(f'all assets will be saved in {target_dir}')

    audio_path = join(target_dir, 'audio.wav')
    transcript_path = join(target_dir, 'transcript.txt')
    asr_path = join(target_dir, 'asr.pkl')
    alignment_json_path = join(target_dir, 'alignment.json')

    print(f'saving audio in {audio_path}')
    write_wav_file(audio_path, audio, rate)
    print(f'saving transcript in {transcript_path}')
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)

    if exists(asr_path):
        print(f'VAD + ASR: loading cached results from pickle: {asr_path}')
        with open(asr_path, 'rb') as f:
            voice_activities = pickle.load(f)
    else:
        print(f'VAD: Splitting audio into speech segments')
        voice_activities = extract_voice_activities(audio, rate, limit)
        print(f'ASR: transcribing each segment')
        voice_activities = transcribe(voice_activities, language, printout=True)
        print(f'saving results to {asr_path}')
        with open(asr_path, 'wb') as f:
            pickle.dump(voice_activities, f)

    print(f'aligning audio with transcript')
    alignment = align(voice_activities, transcript.replace('\n', ' '), printout=True)

    print(f'saving alignment information to {alignment_json_path}')
    json_data = create_alignment_json(alignment)
    with open(alignment_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    create_demo_index(target_dir, sample_id, transcript)
    update_index(sample_id)


def create_alignment_json(alignments):
    words = []
    for al in alignments:
        start_ms = frame_to_ms(al.start_frame, al.rate) * 2
        end_ms = frame_to_ms(al.end_frame, al.rate) * 2
        words.append([al.alignment_text, start_ms, end_ms])

    json_data = {}
    json_data['words'] = words
    return json_data


def create_demo_index(target_dir, sample_id, transcript):
    template_path = join(DEMO_ROOT, '_template.html')
    soup = BeautifulSoup(open(template_path), 'html.parser')
    soup.title.string = sample_id
    soup.find(id='demo_title').string = f'Forced Alignment for {sample_id}'
    soup.find(id='target').string = transcript

    demo_html = join(target_dir, 'index.html')
    with open(demo_html, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())

    return demo_html


def update_index(sample_id):
    index_path = join(DEMO_ROOT, 'index.html')
    soup = BeautifulSoup(open(index_path), 'html.parser')

    if not soup.find(id=sample_id):
        a = soup.new_tag('a', href=sample_id)
        a.string = sample_id
        li = soup.new_tag('li', id=sample_id)
        li.append(a)
        ul = soup.find(id='demo_list')
        ul.append(li)

        with open(index_path, 'w') as f:
            f.write(soup.prettify())
