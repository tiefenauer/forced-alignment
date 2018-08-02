# Create ReadyLingua Corpus
import argparse
import logging
import math
import os
import re
import sys
from os import makedirs, walk
from os.path import exists, splitext, join
from pathlib import Path

from librosa.output import write_wav
from pydub.utils import mediainfo
from tqdm import tqdm

from constants import CORPUS_ROOT, CORPUS_RAW_ROOT
from corpus.corpus import LibriSpeechCorpus
from corpus.corpus_entry import CorpusEntry
from corpus.corpus_segment import Speech, Pause, UnalignedSpeech
from util.audio_util import crop_to_segments, seconds_to_frame, read_audio
from util.corpus_util import save_corpus, find_file_by_suffix
from util.log_util import log_setup, create_args_str

logfile = 'create_ls_corpus.log'
log_setup(filename=logfile)
log = logging.getLogger(__name__)

# -------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="""Create LibriSpeech corpus from raw files""")
parser.add_argument('-f', '--file', help='Dummy argument for Jupyter Notebook compatibility')
parser.add_argument('-s', '--source_root', default=CORPUS_RAW_ROOT,
                    help=f'(optional) source root directory (default: {CORPUS_RAW_ROOT}')
parser.add_argument('-t', '--target_root', default=CORPUS_ROOT,
                    help=f'(optional) target root directory (default: {CORPUS_ROOT})')
parser.add_argument('-m', '--max_entries', type=int, default=None,
                    help='(optional) maximum number of corpus entries to process. Default=None=\'all\'')
parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                    help='(optional) overwrite existing audio data if already present. If set to true this will '
                         'convert, resample and crop the audio data to a 16kHz mono WAV file which will prolong the'
                         'corpus creation process considerably. If set to false, the conversion of audio data will be'
                         'skipped, if the file is already present in the target directory and the corpus will only be'
                         'updated with the most current corpus entries. Default=False)')
args = parser.parse_args()

# -------------------------------------------------------------
# Other values
# -------------------------------------------------------------
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

non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
punctuation_pattern = re.compile(r'[^\w\s]')
whitespace_pattern = re.compile(r'\s+')


def main():
    print(create_args_str(args))
    source_root = join(args.source_root, 'librispeech-raw')
    target_root = join(args.target_root, 'librispeech-corpus')
    print(f'Processing files from {source_root} and saving them in {target_root}')

    corpus, corpus_file = create_corpus(source_root, target_root, args.max_entries)

    print(f'Done! Corpus with {len(corpus)} entries saved to {corpus_file}')


def create_corpus(source_root, target_root, max_entries=None):
    if not exists(source_root):
        print(f"ERROR: Source root {source_root} does not exist!")
        exit(0)
    if not exists(target_root):
        makedirs(target_root)

    return create_librispeech_corpus(source_root=source_root, target_root=target_root, max_entries=max_entries)


def create_librispeech_corpus(source_root, target_root, max_entries):
    audio_root = join(source_root, 'audio')
    book_info, chapter_info, speaker_info = collect_corpus_info(audio_root)

    print('loading book texts')
    books_root = join(source_root, 'books')
    books = collect_book_texts(books_root)

    print('creating corpus entries')
    corpus_entries = []

    directories = [root for root, subdirs, files in walk(audio_root) if not subdirs]
    progress = tqdm(directories, total=min(len(directories), max_entries or math.inf), file=sys.stderr, unit='entries')

    for raw_path in progress:
        if max_entries and len(corpus_entries) >= max_entries:
            break

        progress.set_description(f'{raw_path:{100}}')

        parms = collect_corpus_entry_parms(raw_path, book_info, chapter_info, speaker_info)

        segments_file, transcript_file, mp3_file = collect_corpus_entry_files(raw_path, parms)
        segments_file = join(raw_path, segments_file)
        transcript_file = join(raw_path, transcript_file)

        if not segments_file or not transcript_file or not mp3_file:
            log.warning(f'Skipping directory (not all files found): {raw_path}')
            break

        book_id = parms['book_id']
        book_text = books[book_id] if book_id in books else ''
        if not book_text:
            log.warning(f'No book text found. Processing directory, but speech pauses might be wrong.')

        segments, full_transcript = create_segments(segments_file, transcript_file, book_text)

        # Convert, resample and crop audio
        audio_file = join(raw_path, mp3_file)
        target_audio_path = join(target_root, splitext(mp3_file)[0] + ".wav")
        if not exists(target_audio_path) or args.overwrite:
            audio, rate = read_audio(audio_file, resample_rate=16000, to_mono=True)
            audio, rate, segments = crop_to_segments(audio, rate, segments)
            write_wav(target_audio_path, audio, rate)
        parms['media_info'] = mediainfo(target_audio_path)

        # Create corpus entry
        corpus_entry = CorpusEntry(target_audio_path, segments, full_transcript=full_transcript, raw_path=raw_path,
                                   parms=parms)
        corpus_entries.append(corpus_entry)

    corpus = LibriSpeechCorpus(corpus_entries, target_root)
    corpus_file = save_corpus(corpus, target_root)
    return corpus, corpus_file


def collect_corpus_info(directory):
    # books
    books_file = find_file_by_suffix(directory, 'BOOKS.TXT')
    books_file = join(directory, books_file)
    books = collect_books(books_file)

    # chapters
    chapters_file = find_file_by_suffix(directory, 'CHAPTERS.TXT')
    chapters_file = join(directory, chapters_file)
    chapters = collect_chapters(chapters_file)

    # speakers
    speakers_file = find_file_by_suffix(directory, 'SPEAKERS.TXT')
    speakers_file = join(directory, speakers_file)
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


def collect_book_texts(books_root):
    book_texts = {}
    for root, files in tqdm([(root, files) for root, subdirs, files in walk(books_root)
                             if not subdirs and len(files) == 1], unit='books'):
        book_path = join(root, files[0])
        encoding = 'latin-1' if 'ascii' in book_path else 'utf-8'  # use latin-1 for ascii files because of encoding problems

        book_id = root.split(os.sep)[-1]
        book_text = Path(book_path).read_text(encoding=encoding)
        book_texts[book_id] = book_text
    return book_texts


def collect_corpus_entry_parms(directory, book_info, chapter_info, speaker_info):
    files_pattern = re.compile("[\\\/]mp3[\\\/](?P<speaker_id>\d*)[\\\/](?P<chapter_id>\d*)")
    result = re.search(files_pattern, directory)
    if result:
        speaker_id = result.group('speaker_id')
        chapter_id = result.group('chapter_id')

        chapter = chapter_info[chapter_id] if chapter_id in chapter_info else {'chapter_title': 'unknown',
                                                                               'book_id': 'unknown',
                                                                               'subset': 'unknown'}
        speaker = speaker_info[speaker_id] if speaker_id in speaker_info else 'unknown'

        book_id = chapter['book_id']

        book_title = book_info[book_id] if book_id in book_info else chapter['project_title']
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
    segments_file = find_file_by_suffix(directory, f'{speaker_id}-{chapter_id}.seg.txt')
    transcript_file = find_file_by_suffix(directory, f'{speaker_id}-{chapter_id}.trans.txt')
    mp3_file = find_file_by_suffix(directory, f'{chapter_id}.mp3')
    return segments_file, transcript_file, mp3_file


def normalize_text(text):
    text = text.upper()
    text = text.replace('-', ' ')
    text = re.sub(non_ascii_pattern, '', text)
    text = re.sub(punctuation_pattern, '', text)
    text = re.sub(whitespace_pattern, ' ', text)
    return text


def find_text_between(prev_text, next_text, book_text):
    prev_text = normalize_text(prev_text)
    next_text = normalize_text(next_text)

    if prev_text in book_text and next_text in book_text:
        # find occurrences of prev_text and nex_text which are closes to each other
        prev_indices = [(m.start(), m.end()) for m in re.finditer(prev_text, book_text)]
        min_distance = math.inf
        start = end = 0
        for prev_start, prev_end in prev_indices:
            next_indices = [(m.start(), m.end()) for m in re.finditer(next_text, book_text) if m.start() > prev_end]
            for next_start, next_end in next_indices:
                distance = next_start - prev_end
                if distance < min_distance:
                    min_distance = distance
                    start = prev_end + 1
                    end = next_start - 1

        between_text = book_text[start:end]
        return between_text
    return None


def create_segments(segments_file, transcript_file, book_text):
    book_text = normalize_text(book_text)
    full_transcript = ''

    segment_texts = {}
    with open(transcript_file, 'r') as f_transcript:
        for line in f_transcript.readlines():
            segment_id, segment_text = line.split(' ', 1)
            segment_texts[segment_id] = segment_text.replace('\n', '')

    segments = []
    with open(segments_file, 'r') as f_segments:
        lines = f_segments.readlines()
        for i, line in enumerate(lines):
            segment_id, next_start, next_end = parse_segment_line(line)
            segment_text = segment_texts[segment_id] if segment_id in segment_texts else ''
            full_transcript += segment_text + '\n'

            # add pause or missing speech segment between speeches (if there is one)
            if i > 0:
                prev_id, prev_start, prev_end = parse_segment_line(lines[i - 1])
                prev_text = segment_texts[prev_id] if prev_id in segment_texts else None
                between_text = find_text_between(prev_text, segment_text, book_text)

                between_start = prev_end + 1
                between_end = next_start - 1
                if between_end - between_start > 0:
                    if between_text:
                        full_transcript += between_text + '\n'
                        between_segment = UnalignedSpeech(start_frame=between_start, end_frame=between_end,
                                                          transcript=between_text)
                    else:
                        between_segment = Pause(start_frame=between_start, end_frame=between_end)
                    segments.append(between_segment)

            speech = Speech(start_frame=next_start, end_frame=next_end, transcript=segment_text)
            segments.append(speech)

    return segments, full_transcript


def parse_segment_line(line):
    result = re.search(segment_pattern, line)
    if result:
        segment_id = result.group('segment_id')
        segment_start = seconds_to_frame(result.group('segment_start'))
        segment_end = seconds_to_frame(result.group('segment_end'))
        return segment_id, segment_start, segment_end
    return None, None, None


if __name__ == '__main__':
    main()
