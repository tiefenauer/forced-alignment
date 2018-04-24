# Create ReadyLingua Corpus
import logging
import math
import os
import sys
import wave
from pathlib import Path

from lxml import etree
from tqdm import tqdm

from audio_util import recalculate_frame, resample_wav
from corpus import CorpusEntry, Alignment, Segment, Corpus
from corpus_util import save_corpus, find_file_by_extension
from util import log_setup

logfile = 'readylingua_corpus.log'
log_setup(filename=logfile)
log = logging.getLogger(__name__)

SOURCE_ROOT = r'D:\corpus\readylingua-raw'
TARGET_ROOT = r'E:\readylingua-corpus'
LANGUAGES = {
    'Deutsch': 'de',
    'Englisch': 'en',
    'Französisch': 'fr',
    'Italienisch': 'it',
    'Spanisch': 'es'
}


def create_corpus(source_root=SOURCE_ROOT, target_root=TARGET_ROOT, max_entries=None):
    if not os.path.exists(source_root):
        print(f"ERROR: Source root {source_root} does not exist!")
        exit(0)
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    return create_readylingua_corpus(source_root=source_root, target_root=target_root, max_entries=max_entries)


def create_readylingua_corpus(source_root=SOURCE_ROOT, target_root=TARGET_ROOT, max_entries=None):
    """ Iterate through all leaf directories that contain the audio and the alignment files """
    log.info('Collecting files')
    corpus_entries = []

    directories = [root for root, subdirs, files in os.walk(source_root) if not subdirs]
    progress = tqdm(directories, total=min(len(directories), max_entries or math.inf), file=sys.stderr)

    for directory in progress:
        if max_entries and len(corpus_entries) >= max_entries:
            break

        progress.set_description(f'{directory:{100}}')

        files = collect_files(directory)
        if not files:
            log.warning(f'Skipping directory (not all files found): {directory}')
            continue

        parms = collect_corpus_entry_parms(directory, files)

        # Downsample audio
        wav_file = files['audio']
        src = os.path.join(directory, wav_file)
        dst = os.path.join(target_root, wav_file.split(".")[0] + "_16.wav")
        audio_file = resample_wav(src, dst, inrate=parms['rate'], inchannels=parms['channels'])

        # Calculate speech pauses
        segmentation_file = os.path.join(directory, files['segmentation'])
        speech_pauses = create_segments(segmentation_file)

        # Calculate alignments
        transcript = Path(directory, files['text']).read_text(encoding='utf-8')
        index_file = os.path.join(directory, files['index'])
        transcript, alignments = create_alignments(transcript, index_file)

        # Create corpus entry
        corpus_entry = CorpusEntry(audio_file, transcript, alignments, speech_pauses, directory, parms)
        corpus_entries.append(corpus_entry)

    corpus = Corpus('LibriSpeech', corpus_entries)
    corpus_file = os.path.join(target_root, 'readylingua.corpus')
    save_corpus(corpus, corpus_file)
    print(f'Corpus files saved to {corpus_file}')
    return corpus_file


def collect_files(directory):
    project_file = find_file_by_extension(directory, ' - Project.xml')
    if project_file:
        audio_file, text_file, segmentation_file, index_file = parse_project_file(os.path.join(directory, project_file))
    else:
        audio_file, text_file, segmentation_file, index_file = scan_content_dir(directory)

    files = {'audio': audio_file, 'text': text_file, 'segmentation': segmentation_file, 'index': index_file}

    # check if all file names were found
    for attribute_name, file_name in files.items():
        if not file_name:
            log.warning(f'File \'{attribute_name}\' is not set')
            return None

    # check if files exist
    for file_name in files.values():
        path = Path(directory, file_name)
        if not path.exists():
            log.warning(f'File does not exist: {path}')
            return None

    return files


def parse_project_file(project_file):
    doc = etree.parse(project_file)
    for element in ['AudioFiles/Name', 'TextFiles/Name', 'SegmentationFiles/Name', 'IndexFiles/Name']:
        if doc.find(element) is None:
            log.warning(f'Invalid project file (missing element \'{element}\'): {project_file}')
            return None, None, None, None

    audio_file = doc.find('AudioFiles/Name').text
    text_file = doc.find('TextFiles/Name').text
    segmentation_file = doc.find('SegmentationFiles/Name').text
    index_file = doc.find('IndexFiles/Name').text
    return audio_file, text_file, segmentation_file, index_file


def scan_content_dir(content_dir):
    audio_file = find_file_by_extension(content_dir, '.wav')
    text_file = find_file_by_extension(content_dir, '.txt')
    segmentation_file = find_file_by_extension(content_dir, ' - Segmentation.xml')
    index_file = find_file_by_extension(content_dir, ' - Index.xml')
    return audio_file, text_file, segmentation_file, index_file


def collect_corpus_entry_parms(directory, files):
    index_file_path = os.path.join(directory, files['index'])
    audio_file_path = os.path.join(directory, files['audio'])
    folders = directory.split('\\')

    # find name
    name = folders[-1]

    # find language
    lang = [folder for folder in folders if folder in LANGUAGES.keys()]
    language = LANGUAGES[lang[0]] if lang else 'unknown'

    # find sampling rate
    doc = etree.parse(index_file_path)
    rate = int(doc.find('SamplingRate').text)

    # find number of channels
    channels = wave.open(audio_file_path, 'rb').getnchannels()

    return {'name': name, 'language': language, 'rate': rate, 'channels': channels}


def create_segments(segmentation_file):
    """ re-calculate speech pauses for downsampled WAV file """
    segments = []
    doc = etree.parse(segmentation_file)
    for element in doc.findall('Segments/Segment'):
        # calculate new start and end frame positions
        start_frame = recalculate_frame(int(element.attrib['start']))
        end_frame = recalculate_frame(int(element.attrib['end']))

        segment = Segment(start_frame, end_frame, element.attrib['class'])
        segments.append(segment)

    return segments


def create_alignments(text, index_file):
    alignments = []
    doc = etree.parse(index_file)
    for element in doc.findall('TextAudioIndex'):
        start_text = int(element.find('TextStartPos').text)
        end_text = int(element.find('TextEndPos').text)
        audio_start = int(element.find('AudioStartPos').text)
        audio_end = int(element.find('AudioEndPos').text)
        start_frame = recalculate_frame(audio_start)
        end_frame = recalculate_frame(audio_end)

        alignment = Alignment(start_frame, end_frame, start_text, end_text)
        alignments.append(alignment)

    return text, alignments


if __name__ == '__main__':
    create_corpus()
