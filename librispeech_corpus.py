# Create ReadyLingua Corpus
import argparse
import logging
import math
import os
import re
import sys

from os.path import exists
from pydub.utils import mediainfo
from tqdm import tqdm

from audio_util import mp3_to_wav, crop_wav, calculate_frame
from corpus import Pause, CorpusEntry, LibriSpeechCorpus, Speech
from corpus_util import save_corpus, find_file_by_extension
from util import log_setup

logfile = 'librispeech_corpus.log'
log_setup(filename=logfile)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="""Create LibriSpeech corpus from raw files""")
parser.add_argument('-f', '--file', help='Dummy argument for Jupyter Notebook compatibility')
parser.add_argument('-m', '--max_entries', type=int, default=None,
                    help='(optional) maximum number of corpus entries to process. Default=None=\'all\'')
parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                    help='(optional) overwrite existing audio data if already present. Default=False)')
args = parser.parse_args()

books_pattern = re.compile('(?P<book_id>\d+)'
                           '\s*\|\s*'
                           '(?P<book_title>.*?)'
                           '\s*(\|\s*|\n)')
speakers_pattern = re.compile('(?P<speaker_id>\d+)'
                              '\s*\|\s*'
                              '(?P<sex>[MF])'
                              '\s*\|\s*'
                              '(?P<subset>.*?)'
                              '\s*\|\s*'
                              '(?P<minutes>\d[\d.]*)'
                              '\s*\|\s*'
                              '(?P<speaker_name>.*)')
chapters_pattern = re.compile("(?P<chapter_id>\d+)"
                              "\s*\|\s*"
                              "(?P<reader_id>\d+)"
                              "\s*\|\s*"
                              "(?P<minutes>\d[\d.]*)"
                              "\s*\|\s*"
                              "(?P<subset>.*?)"
                              "\s*\|\s*"
                              "(?P<project_id>\d+)"
                              "\s*\|\s*"
                              "(?P<book_id>\d+)"
                              "\s*\|\s*"
                              "(?P<chapter_title>.*)"
                              "\s*\|\s*"
                              "(?P<project_title>.*)")
segment_pattern = re.compile('(?P<segment_id>.*)\s(?P<segment_start>.*)\s(?P<segment_end>.*)\n')

source_root = r'D:\corpus\librispeech-raw' if os.name == 'nt' else '/media/all/D1/librispeech-raw'
target_root = r'E:\librispeech-corpus' if os.name == 'nt' else '/media/all/D1/librispeech-corpus'
max_entries = args.max_entries
overwrite = args.overwrite


def create_corpus(source_root=source_root, target_root=target_root, max_entries=max_entries):
    if not os.path.exists(source_root):
        print(f"ERROR: Source root {source_root} does not exist!")
        exit(0)
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    return create_librispeech_corpus(source_root=source_root, target_root=target_root, max_entries=max_entries)


def create_librispeech_corpus(source_root, target_root, max_entries):
    audio_root = os.path.join(source_root, 'audio')
    books_root = os.path.join(source_root, 'books')
    books, chapters, speakers = collect_corpus_info(audio_root)
    corpus_entries = []

    directories = [root for root, subdirs, files in os.walk(audio_root) if not subdirs]
    progress = tqdm(directories, total=min(len(directories), max_entries or math.inf), file=sys.stderr, unit='entries')

    for directory in progress:
        if max_entries and len(corpus_entries) >= max_entries:
            break

        progress.set_description(f'{directory:{100}}')

        parms = collect_corpus_entry_parms(directory, books, chapters, speakers)

        segments_file, transcription_file, mp3_file = collect_corpus_entry_files(directory, parms)
        segments_file = os.path.join(directory, segments_file)
        transcription_file = os.path.join(directory, transcription_file)

        if not segments_file or not transcription_file or not mp3_file:
            log.warning(f'Skipping directory (not all files found): {directory}')
            break

        segments, transcript = create_segments(segments_file, transcription_file)

        # Convert, resample and crop audio
        audio_file = os.path.join(target_root, mp3_file.split(".")[0] + "_16.wav")
        if not exists(audio_file) or overwrite:
            in_file = os.path.join(directory, mp3_file)
            mp3_to_wav(in_file, audio_file)
            crop_wav(audio_file, segments)
            parms['media_info'] = mediainfo(audio_file)

        # Create corpus entry
        corpus_entry = CorpusEntry(audio_file, transcript, segments, directory, parms)
        corpus_entries.append(corpus_entry)

    corpus = LibriSpeechCorpus(corpus_entries, target_root)
    corpus_file = os.path.join(target_root, 'librispeech.corpus')
    save_corpus(corpus, corpus_file)
    return corpus, corpus_file


def collect_corpus_info(directory):
    # books
    books_file = find_file_by_extension(directory, 'BOOKS.TXT')
    books_file = os.path.join(directory, books_file)
    books = collect_books(books_file)

    # chapters
    chapters_file = find_file_by_extension(directory, 'CHAPTERS.TXT')
    chapters_file = os.path.join(directory, chapters_file)
    chapters = collect_chapters(chapters_file)

    # speakers
    speakers_file = find_file_by_extension(directory, 'SPEAKERS.TXT')
    speakers_file = os.path.join(directory, speakers_file)
    speakers = collect_speakers(speakers_file)

    return books, chapters, speakers


def collect_info(file, pattern):
    with open(file) as f:
        for line in (line for line in f.readlines() if not line.startswith(';')):
            results = re.search(pattern, line)
            if results:
                yield results


def collect_books(books_file):
    books = {}
    for result in collect_info(books_file, books_pattern):
        book_id = result.group('book_id') if result.group('book_id') else 'unknown'
        book_title = result.group('book_title') if result.group('book_title') else 'unknown'
        books[book_id] = book_title
    books['unknown'] = 'unknown'
    return books


def collect_chapters(chapters_file):
    chapters = {}
    for result in collect_info(chapters_file, chapters_pattern):
        chapter_id = result.group('chapter_id')
        chapter = {
            'reader_id': result.group('reader_id'),
            'length': float(result.group('minutes')),
            'subset': result.group('subset'),
            'project_id': result.group('project_id'),
            'book_id': result.group('book_id'),
            'chapter_title': result.group('chapter_title'),
            'project_title': result.group('project_title')
        }
        chapters[chapter_id] = chapter
    chapters['unknown'] = 'unknown'
    return chapters


def collect_speakers(speakers_file):
    speakers = {}
    for result in collect_info(speakers_file, speakers_pattern):
        speaker_id = result.group('speaker_id')
        speaker = {
            'sex': result.group('sex'),
            'subset': result.group('subset'),
            'length': float(result.group('minutes')),
            'name': result.group('speaker_name'),
        }
        speakers[speaker_id] = speaker
    speakers['unknown'] = 'unknown'
    return speakers


def collect_corpus_entry_parms(directory, books, chapters, speakers):
    files_pattern = re.compile("[\\\/]mp3[\\\/](?P<speaker_id>\d*)[\\\/](?P<chapter_id>\d*)")
    result = re.search(files_pattern, directory)
    if result:
        speaker_id = result.group('speaker_id')
        chapter_id = result.group('chapter_id')

        chapter = chapters[chapter_id] if chapter_id in chapters else {'chapter_title': 'unknown', 'book_id': 'unknown',
                                                                       'subset': 'unknown'}
        speaker = speakers[speaker_id] if speaker_id in speakers else 'unknown'

        book_id = chapter['book_id']

        book_title = books[book_id] if book_id in books else chapter['project_title']
        chapter_title = chapter['chapter_title']
        subset = chapter['subset']
        speaker_name = speaker['name']

        return {'name': book_title,
                'id': chapter_id,
                'chapter_title': chapter_title,
                'language': 'en',
                'book_id': book_id,
                'speaker_id': speaker_id,
                'chapter_id': chapter_id,
                'speaker_name': speaker_name,
                'subset': subset}


def collect_corpus_entry_files(directory, parms):
    speaker_id = parms['speaker_id']
    chapter_id = parms['chapter_id']
    segments_file = find_file_by_extension(directory, f'{speaker_id}-{chapter_id}.seg.txt')
    transcription_file = find_file_by_extension(directory, f'{speaker_id}-{chapter_id}.trans.txt')
    mp3_file = find_file_by_extension(directory, f'{chapter_id}.mp3')
    return segments_file, transcription_file, mp3_file


def create_segments(segments_file, transcription_file):
    segment_texts = {}
    with open(transcription_file) as f_transcription:
        for line in f_transcription.readlines():
            segment_id, segment_text = line.split(' ', 1)
            segment_texts[segment_id] = segment_text.replace('\n', '')
    transcription = '\n'.join(segment_texts.values())

    segments = []
    with open(segments_file) as f_segments:
        lines = f_segments.readlines()
        for i, line in enumerate(lines):
            segment_id, start_frame, end_frame = parse_segment_line(line)

            # add pause between speeches (if there is one)
            if i > 0:
                _, prev_start, prev_end = parse_segment_line(lines[i - 1])
                pause_start = prev_end + 1
                pause_end = start_frame - 1
                if pause_end - pause_start > 0:
                    pause = Pause(start_frame=pause_start, end_frame=pause_end)
                    segments.append(pause)

            segment_text = segment_texts[segment_id] if segment_id in segment_texts else None
            start_text = transcription.index(segment_text)
            end_text = start_text + len(segment_text)
            speech = Speech(start_frame=start_frame, end_frame=end_frame, start_text=start_text, end_text=end_text)
            segments.append(speech)

    return segments, transcription


def parse_segment_line(line):
    result = re.search(segment_pattern, line)
    if result:
        segment_id = result.group('segment_id')
        segment_start = calculate_frame(result.group('segment_start'))
        segment_end = calculate_frame(result.group('segment_end'))
        return segment_id, segment_start, segment_end
    return None, None, None


if __name__ == '__main__':
    print(f'source_root={source_root}, target_root={target_root}, max_entries={max_entries}, overwrite={overwrite}')
    corpus, corpus_file = create_corpus(source_root, target_root, max_entries)
    print(f'Done! Corpus with {len(corpus)} entries saved to {corpus_file}')
