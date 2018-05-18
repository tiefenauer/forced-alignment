# Create ReadyLingua Corpus
import argparse
import logging
import math
import os
import sys
import wave
from pathlib import Path

from lxml import etree
from os.path import exists
from pydub.utils import mediainfo
from tqdm import tqdm

from audio_util import recalculate_frame, resample_wav, crop_wav
from corpus import CorpusEntry, ReadyLinguaCorpus, Speech, Pause
from corpus_util import save_corpus, find_file_by_extension
from log_util import log_setup
from string_utils import create_filename

logfile = 'readylingua_corpus.log'
log_setup(filename=logfile)
log = logging.getLogger(__name__)

DEFAULT_SOURCE_ROOT = r'D:\\corpus\\' if os.name == 'nt' else '/media/all/D1/'
DEFAULT_TARGET_ROOT = r'D:\\corpus\\' if os.name == 'nt' else '/media/all/D1/'

parser = argparse.ArgumentParser(description="""Create LibriSpeech corpus from raw files""")
parser.add_argument('-f', '--file', help='Dummy argument for Jupyter Notebook compatibility')
parser.add_argument('-s', '--source_root', default=DEFAULT_SOURCE_ROOT,
                    help=f'(optional) source root directory (default: {DEFAULT_SOURCE_ROOT}')
parser.add_argument('-t', '--target_root', default=DEFAULT_TARGET_ROOT,
                    help=f'(optional) target root directory (default: {DEFAULT_TARGET_ROOT})')
parser.add_argument('-m', '--max_entries', type=int, default=None,
                    help='(optional) maximum number of corpus entries to process. Default=None=\'all\'')
parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                    help='(optional) overwrite existing audio data if already present. Default=False)')
args = parser.parse_args()

max_entries = args.max_entries
overwrite = args.overwrite

LANGUAGES = {
    'Deutsch': 'de',
    'Englisch': 'en',
    'Französisch': 'fr',
    'Italienisch': 'it',
    'Spanisch': 'es'
}

source_root = os.path.join(args.source_root, 'readylingua-raw')
target_root = os.path.join(args.target_root, 'readylingua-corpus')


def create_corpus(source_root=source_root, target_root=target_root, max_entries=max_entries):
    if not os.path.exists(source_root):
        print(f"ERROR: Source root {source_root} does not exist!")
        exit(0)
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    return create_readylingua_corpus(source_root, target_root, max_entries)


def create_readylingua_corpus(source_root, target_root, max_entries):
    """ Iterate through all leaf directories that contain the audio and the alignment files """
    log.info('Collecting files')
    corpus_entries = []

    directories = [root for root, subdirs, files in os.walk(source_root)
                   if not subdirs  # only include leaf directories
                   and not root.endswith(os.sep + 'old')  # '/old' leaf-folders are considered not reliable
                   and not os.sep + 'old' + os.sep in root]  # also exclude /old/ non-leaf folders

    progress = tqdm(directories, total=min(len(directories), max_entries or math.inf), file=sys.stderr, unit='entries')

    for directory in progress:
        if max_entries and len(corpus_entries) >= max_entries:
            break

        progress.set_description(f'{directory:{100}}')

        files = collect_files(directory)
        if not files:
            log.warning(f'Skipping directory (not all files found): {directory}')
            continue

        parms = collect_corpus_entry_parms(directory, files)

        segmentation_file = os.path.join(directory, files['segmentation'])
        index_file = os.path.join(directory, files['index'])
        transcript_file = os.path.join(directory, files['text'])
        segments = create_segments(index_file, transcript_file, segmentation_file)

        # Resample and crop audio
        wav_file = files['audio']
        audio_file = os.path.join(target_root, parms['id'] + ".wav")
        if not exists(audio_file) or overwrite:
            in_file = os.path.join(directory, wav_file)
            resample_wav(in_file, audio_file, inrate=parms['rate'], inchannels=parms['channels'])
            crop_wav(audio_file, segments)
        parms['media_info'] = mediainfo(audio_file)

        # Create corpus entry
        corpus_entry = CorpusEntry(audio_file, segments, directory, parms)
        corpus_entries.append(corpus_entry)

    corpus = ReadyLinguaCorpus(corpus_entries, target_root)
    corpus_file = os.path.join(target_root, 'readylingua.corpus')
    save_corpus(corpus, corpus_file)
    return corpus, corpus_file


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
    folders = directory.split(os.sep)

    # find name and create id
    corpus_entry_name = folders[-1]
    corpus_entry_id = create_filename(files['audio'].split('.wav')[0])

    # find language
    lang = [folder for folder in folders if folder in LANGUAGES.keys()]
    language = LANGUAGES[lang[0]] if lang else 'unknown'

    # find sampling rate
    doc = etree.parse(index_file_path)
    rate = int(doc.find('SamplingRate').text)

    # find number of channels
    channels = wave.open(audio_file_path, 'rb').getnchannels()

    return {'id': corpus_entry_id, 'name': corpus_entry_name, 'language': language, 'rate': rate, 'channels': channels}


def find_speech_within_segment(segment, speeches):
    return next(iter([speech for speech in speeches
                      if speech['start_frame'] >= segment['start_frame']
                      and speech['end_frame'] <= segment['end_frame']]), None)


def create_segments(index_file, transcription_file, segmentation_file):
    segmentation = collect_segmentation(segmentation_file)
    speeches = collect_speeches(index_file)
    transcription = Path(transcription_file).read_text(encoding='utf-8')

    # merge information from index file (speech parts) with segmentation information
    segments = []
    for audio_segment in segmentation:
        start_frame = audio_segment['start_frame']
        end_frame = audio_segment['end_frame']
        if audio_segment['class'] == 'Speech':
            speech = find_speech_within_segment(audio_segment, speeches)
            if speech:
                start_text = speech['start_text']
                end_text = speech['end_text'] + 1  # komische Indizierung
                seg_transcription = transcription[start_text:end_text]
                segments.append(Speech(start_frame=start_frame, end_frame=end_frame, transcription=seg_transcription))
        else:
            segments.append(Pause(start_frame=start_frame, end_frame=end_frame))

    return segments


def collect_segmentation(segmentation_file):
    segments = []
    doc = etree.parse(segmentation_file)
    for element in doc.findall('Segments/Segment'):
        start_frame = recalculate_frame(int(element.attrib['start']))
        end_frame = recalculate_frame(int(element.attrib['end']))
        segment = {'class': element.attrib['class'], 'start_frame': start_frame, 'end_frame': end_frame}
        segments.append(segment)

    return sorted(segments, key=lambda s: s['start_frame'])


def collect_speeches(index_file):
    speeches = []
    doc = etree.parse(index_file)
    for element in doc.findall('TextAudioIndex'):
        start_text = int(element.find('TextStartPos').text)
        end_text = int(element.find('TextEndPos').text)
        start_frame = int(element.find('AudioStartPos').text)
        end_frame = int(element.find('AudioEndPos').text)
        start_frame = recalculate_frame(start_frame)
        end_frame = recalculate_frame(end_frame)

        speech = {'start_frame': start_frame, 'end_frame': end_frame, 'start_text': start_text, 'end_text': end_text}
        speeches.append(speech)
    return sorted(speeches, key=lambda s: s['start_frame'])


if __name__ == '__main__':
    print(f'source_root={source_root}, target_root={target_root}, max_entries={max_entries}, overwrite={overwrite}')
    corpus, corpus_file = create_corpus(source_root, target_root, max_entries)
    print(f'Done! Corpus with {len(corpus)} entries saved to {corpus_file}')
