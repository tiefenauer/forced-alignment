# Create ReadyLingua Corpus
import logging
import math
import os
import re
import sys
from operator import itemgetter

from tqdm import tqdm

from audio_util import calculate_frame, mp3_to_wav
from corpus import Corpus, Alignment, Segment, CorpusEntry
from corpus_util import save_corpus, find_file_by_extension
from util import log_setup

logfile = 'librispeech_corpus.log'
log_setup(filename=logfile)
log = logging.getLogger(__name__)

SOURCE_ROOT = r'D:\corpus\librispeech-raw\audio'  # location of raw audio files
TARGET_ROOT = r'E:\librispeech-corpus'  # target location of corpus

books_pattern = re.compile('(?P<book_id>\d+)'
                           '\s*\|\s*'
                           '(?P<book_title>.*?)'
                           '\s*\|\s*')
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


def create_corpus(source_root=SOURCE_ROOT, target_root=TARGET_ROOT, max_entries=None):
    if not os.path.exists(source_root):
        print(f"ERROR: Source root {source_root} does not exist!")
        exit(0)
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    return create_librispeech_corpus(source_root=source_root, target_root=target_root, max_entries=max_entries)


def create_librispeech_corpus(source_root=SOURCE_ROOT, target_root=TARGET_ROOT, max_entries=None):
    books, chapters, speakers = collect_corpus_info(source_root)
    corpus_entries = []

    directories = [root for root, subdirs, files in os.walk(source_root) if not subdirs]
    progress = tqdm(directories, total=min(len(directories), max_entries or math.inf), file=sys.stderr)

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

        # Downsample audio
        infile = os.path.join(directory, mp3_file)
        outfile = os.path.join(target_root, mp3_file.split(".")[0] + "_16.wav")
        audio_file = mp3_to_wav(infile, outfile)

        # create segments
        speech_segments, transcript = create_speech_segments(segments_file, transcription_file)

        # create alignments
        alignments = create_alignments(speech_segments, transcript)

        # create speech pauses
        speech_pauses = create_speech_pauses(speech_segments)

        # Create corpus entry
        corpus_entry = CorpusEntry(audio_file, transcript, alignments, speech_pauses, directory, parms)
        corpus_entries.append(corpus_entry)

    corpus = Corpus('LibriSpeech', corpus_entries)
    corpus_file = os.path.join(target_root, 'librispeech.corpus')
    save_corpus(corpus, corpus_file)
    return corpus_file


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

        chapter = chapters[chapter_id] if chapter_id in chapters else {'chapter_title': 'unknown', 'book_id': 'unknown'}
        book_id = chapter['book_id']

        speaker = speakers[speaker_id] if speaker_id in speakers else 'unknown'
        book_title = books[book_id] if book_id in books else 'unknown'

        return {'name': book_title,
                'chapter_title': chapter['chapter_title'],
                'language': 'en',
                'book_id': book_id,
                'speaker_id': speaker_id,
                'chapter_id': chapter_id,
                'speaker_name': speaker['name']}


def collect_corpus_entry_files(directory, parms):
    speaker_id = parms['speaker_id']
    chapter_id = parms['chapter_id']
    segments_file = find_file_by_extension(directory, f'{speaker_id}-{chapter_id}.seg.txt')
    transcription_file = find_file_by_extension(directory, f'{speaker_id}-{chapter_id}.trans.txt')
    mp3_file = find_file_by_extension(directory, f'{chapter_id}.mp3')
    return segments_file, transcription_file, mp3_file


def create_speech_segments(segments_file, transcription_file):
    segments = []
    segment_texts = {}
    with open(segments_file) as f_segments, open(transcription_file) as f_transcription:
        for line in f_transcription.readlines():
            segment_id, segment_text = line.split(' ', 1)
            segment_texts[segment_id] = segment_text.replace('\n', '')

        for line in f_segments.readlines():
            result = re.search(segment_pattern, line)
            if result:
                segment_id = result.group('segment_id')
                segment_start = result.group('segment_start')
                segment_end = result.group('segment_end')

                segment_text = segment_texts[segment_id] if segment_id in segment_texts else None

                if segment_text:
                    start_frame = calculate_frame(segment_start)
                    end_frame = calculate_frame(segment_end)
                    segment = {'id': segment_id,
                               'start_frame': start_frame,
                               'end_frame': end_frame,
                               'text': segment_text}
                    segments.append(segment)
    transcription = '\n'.join(segment_texts.values())
    return segments, transcription


def create_alignments(segments, transcription):
    alignments = []
    segments = sorted(segments, key=itemgetter('start_frame'))
    for segment in segments:
        segment_text = segment['text']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        start_text = transcription.index(segment_text)
        end_text = start_text + len(segment_text)
        alignment = Alignment(start_frame=start_frame, end_frame=end_frame, start_text=start_text, end_text=end_text)
        alignments.append(alignment)
    return alignments


def create_speech_pauses(speech_segments):
    speech_pause_segments = []
    for i, speech in enumerate(speech_segments):
        if i > 0:
            prev_speech = speech_segments[i-1]
            start_frame = prev_speech['end_frame'] + 1
            end_frame = speech['start_frame'] - 1
            pause_segment = Segment(start_frame=start_frame, end_frame=end_frame, segment_type='pause')
            speech_pause_segments.append(pause_segment)

        start_frame = speech['start_frame']
        end_frame = speech['end_frame']
        speech_segment = Segment(start_frame=start_frame, end_frame=end_frame, segment_type='speech')
        speech_pause_segments.append(speech_segment)
    return speech_pause_segments


if __name__ == '__main__':
    create_corpus()
