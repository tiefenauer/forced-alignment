# Create ReadyLingua Corpus
import logging
import os
from pathlib import Path

from lxml import etree

from audio_util import resample_wav
from util import log_setup

log_setup()
log = logging.getLogger(__name__)

SOURCE_DIR = "D:/corpus/readylingua-original"
TARGET_DIR = "D:/corpus/readylingua"


def find_file_by_extension(directory, extension):
    return next(iter(filename for filename in os.listdir(directory) if filename.lower().endswith(extension.lower())),
                None)


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


def create_segments(wav_file, segments_file):
    pass


def create_readylingua_corpus(corpus_dir=SOURCE_DIR):
    """ Iterate through all leaf directories that contain the audio and the alignment files """
    for directory in (root for root, subdirs, files in os.walk(corpus_dir) if not subdirs):
        files = collect_files(directory)
        if not files:
            log.warning(f'Skipping directory: {directory}')
            continue

        # Downsample audio file to 16kHz
        wav_file = files['audio']
        src = os.path.join(directory, wav_file)
        dst = os.path.join(TARGET_DIR, wav_file.split(".")[0] + "_16.wav")
        # resample_wav(src, dst)

        # create audio segments
        create_segments(dst, files['segments'])

        log.info(f'Processed directory: {directory}')


if __name__ == '__main__':
    if not os.path.exists(SOURCE_DIR):
        log.error("Source directory does not exist!")
        exit(1)
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    create_readylingua_corpus()
