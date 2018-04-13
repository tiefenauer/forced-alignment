# Create ReadyLingua Corpus
import logging
import os
from pathlib import Path
from shutil import copyfile

from lxml import etree
from tqdm import tqdm

from audio_util import calculate_frame
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


def create_speech_pauses(segmentation_file):
    """ re-calculate speech pauses for downsampled WAV file """
    speech_pauses = []
    doc = etree.parse(segmentation_file)
    for element in doc.findall('Segments/Segment'):
        # calculate new start and end frame positions
        start_new = calculate_frame(int(element.attrib['start']))
        end_new = calculate_frame(int(element.attrib['end']))

        speech_pauses.append({'id': element.attrib['id'],
                              'start': start_new,
                              'end': end_new,
                              'class': element.attrib['class']})

        element.attrib['start'] = str(start_new)
        element.attrib['end'] = str(end_new)
    doc.write(segmentation_file, pretty_print=True)
    return speech_pauses


def create_speech_segments(text_file, index_file):
    speech_segments = []
    text = Path(text_file).read_text(encoding='utf-8')
    doc = etree.parse(index_file)
    for element in doc.findall('TextAudioIndex'):
        text_start = int(element.find('TextStartPos').text)
        text_end = int(element.find('TextEndPos').text)
        audio_start = int(element.find('AudioStartPos').text)
        audio_end = int(element.find('AudioEndPos').text)
        audio_start_new = calculate_frame(audio_start)
        audio_end_new = calculate_frame(audio_end)

        text_segment = text[text_start + 1:text_end + 1]  # komische Indizierung...
        speech_segments.append({'text': text_segment, 'start': audio_start_new, 'end': audio_end_new})

        element.find('AudioStartPos').text = str(audio_start_new)
        element.find('AudioEndPos').text = str(audio_end_new)
    doc.write(index_file, pretty_print=True)
    return speech_segments


def create_readylingua_corpus(corpus_dir=SOURCE_DIR):
    """ Iterate through all leaf directories that contain the audio and the alignment files """
    log.info('Collecting files')
    for directory in tqdm(root for root, subdirs, files in os.walk(corpus_dir) if not subdirs):
        files = collect_files(directory)
        if not files:
            log.warning(f'Skipping directory: {directory}')
            continue

        log.info('Downsampling audio')
        wav_file = files['audio']
        src = os.path.join(directory, wav_file)
        dst = os.path.join(TARGET_DIR, wav_file.split(".")[0] + "_16.wav")
        # resample_wav(src, dst)

        log.info('Calculating speech pauses')
        segmentation_file = os.path.join(directory, files['segmentation'])
        segmentation_file = copyfile(segmentation_file, os.path.join(TARGET_DIR, files['segmentation']))
        speech_pauses = create_speech_pauses(segmentation_file)

        log.info('Calculating speech segments')
        text_file = os.path.join(directory, files['text'])
        text_file = copyfile(text_file, os.path.join(TARGET_DIR, files['text']))
        index_file = os.path.join(directory, files['index'])
        index_file = copyfile(index_file, os.path.join(TARGET_DIR, files['index']))
        speech_segments = create_speech_segments(text_file, index_file)


if __name__ == '__main__':
    if not os.path.exists(SOURCE_DIR):
        log.error("Source directory does not exist!")
        exit(1)
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    create_readylingua_corpus()
