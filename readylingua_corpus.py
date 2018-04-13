# Create ReadyLingua Corpus
import audioop
import logging
import os
import sys
import wave
from pathlib import Path

from lxml import etree

from util import log_setup

log_setup()
log = logging.getLogger(__name__)

READYLINGUA_DIR = "D:/corpus/readylingua"


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


def resample_audio(directory, audio_file, inrate=44100, outrate=16000, inchannels=1, outchannels=1):
    """ Downsample WAV file to 16kHz
    Source: https://github.com/rpinsler/deep-speechgen/blob/master/downsample.py
    """
    src = os.path.join(directory, audio_file)
    dst = os.path.join(directory, audio_file.split(".")[0] + "_16.wav")

    try:
        os.remove(dst)
    except OSError:
        pass

    with wave.open(src, 'r') as s_read:
        try:
            n_frames = s_read.getnframes()
            data = s_read.readframes(n_frames)
            converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
            if outchannels == 1 & inchannels != 1:
                converted = audioop.tomono(converted[0], 2, 1, 0)
        except:
            log.error(f'Could not resample audio file {src}: {sys.exc_info()[0]}')

    with wave.open(dst, 'w') as s_write:
        try:
            s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
            s_write.writeframes(converted[0])
        except:
            log.error(f'Could not write resampled data: {dst}')

    return dst


def collect_files(directory):
    project_file = find_file_by_extension(directory, ' - Project.xml')
    if project_file:
        audio_file, text_file, segmentation_file, index_file = parse_project_file(os.path.join(directory, project_file))
    else:
        audio_file, text_file, segmentation_file, index_file = scan_content_dir(directory)

    files = {'audio_file': audio_file,
             'text_file': text_file,
             'segmentation_file': segmentation_file,
             'index_file': index_file}

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


def create_readylingua_corpus(corpus_dir=READYLINGUA_DIR):
    """ Iterate through all leaf directories that contain the audio and the alignment files """
    for leaf_dir in (root for root, subdirs, files in os.walk(corpus_dir) if not subdirs):
        files = collect_files(leaf_dir)
        if not files:
            log.warning(f'Skipping directory: {leaf_dir}')
            continue

        log.info(f'Processed directory: {leaf_dir}')


if __name__ == '__main__':
    create_readylingua_corpus()
