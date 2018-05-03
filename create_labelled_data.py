import argparse
import logging
import math
import os
from os import makedirs

import numpy as np
from os.path import exists
from tqdm import tqdm

from audio_util import log_specgram
from corpus_util import load_corpus
from data_util import save_x
from util import log_setup

logfile = 'create_labelled_data.log'
log_setup(filename=logfile)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Create labelled train-, dev- and test-data (X and Y) for all corpora')
parser.add_argument('-f', '--file', help='Dummy argument for Jupyter Notebook compatibility')
parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                    help='(optional) overwrite existing data if already present. Default=False)')
parser.add_argument('-m', '--max_samples', type=int, default=None,
                    help='(optional) maximum number of samples to process. Default=None=\'all\'')
parser.add_argument('-ty', '--Ty', type=int, default=1375, help='Number of steps in the RNN output layer (T_y)')
args = parser.parse_args()

LS_SOURCE_ROOT = r'E:\librispeech-corpus' if os.name == 'nt' else '/media/all/D1/librispeech-corpus'
RL_SOURCE_ROOT = r'E:\readylingua-corpus' if os.name == 'nt' else '/media/all/D1/readylingua-corpus'
LS_TARGET_ROOT = r'E:\librispeech-data' if os.name == 'nt' else '/media/all/D1/librispeech-data'
RL_TARGET_ROOT = r'E:\readylingua-data' if os.name == 'nt' else '/media/all/D1/readylingua-data'

ls_corpus_file = os.path.join(LS_SOURCE_ROOT, 'librispeech.corpus')
rl_corpus_file = os.path.join(RL_SOURCE_ROOT, 'readylingua.corpus')

# for debugging only: set to a numeric value to limit the amount of processed corpus entries.
overwrite = args.overwrite
max_samples = args.max_samples
T_y = args.Ty


def create_subsets(corpus, target_root):
    if not exists(target_root):
        makedirs(target_root)

    train_set, dev_set, test_set = corpus.train_dev_test_split()

    # limit number of samples per subset
    train_set = train_set[:max_samples] if max_samples else train_set
    dev_set = dev_set[:max_samples] if max_samples else dev_set
    test_set = test_set[:max_samples] if max_samples else test_set

    print('Creating training data...')
    for corpus_entry in tqdm(train_set, total=min(len(train_set), max_samples or math.inf), unit='corpus entry'):
        create_x(corpus_entry, target_root, 'train')
        create_y(corpus_entry, target_root, 'train')
    print('Creating validation data...')
    for corpus_entry in tqdm(dev_set, total=min(len(dev_set), max_samples or math.inf), unit='corpus entry'):
        create_x(corpus_entry, target_root, 'dev')
        create_y(corpus_entry, target_root, 'dev')
    print('Creating test data...')
    for corpus_entry in tqdm(test_set, total=min(len(test_set), max_samples or math.inf), unit='corpus entry'):
        create_x(corpus_entry, target_root, 'test')
        create_y(corpus_entry, target_root, 'test')


def create_x(corpus_entry, target_root, subset_name):
    x_path = os.path.join(target_root, corpus_entry.id + '.X.' + subset_name + '.npy')
    if not exists(x_path) or overwrite:
        rate, audio = corpus_entry.audio
        freqs, times, spec = log_specgram(audio, rate)
        save_x(freqs, times, spec, x_path)
    else:
        print(f'Skipping {x_path} because it already exists')


def create_y(corpus_entry, target_root, subset_name):
    y_path = os.path.join(target_root, corpus_entry.id + '.Y.' + subset_name + '.npy')
    if not exists(y_path) or overwrite:
        duration = float(corpus_entry.media_info['duration'])
        sample_rate = float(corpus_entry.media_info['sample_rate'])
        n_frames = int(duration * sample_rate)
        y = np.zeros(T_y, 'int16')
        for pause_segment in (segment for segment in corpus_entry.segments if segment.segment_type == 'pause'):
            start = round(pause_segment.start_frame * T_y / n_frames)
            end = round(pause_segment.end_frame * T_y / n_frames)
            y[start:end] = 1
        np.save(y_path, y)

        # sum up segment lengths for sanity checks:
        total_len = sum(segment.end_frame - segment.start_frame for segment in corpus_entry.segments)
        difference = abs(n_frames - total_len)
        if difference > n_frames * 0.01:
            msg = f"""Total length of segments ({total_len}) deviated from number of frames in audio ({n_frames}) 
                by {difference/n_frames}%! Training data might contain errors."""
            print(msg)
            log.warning(msg)
    else:
        print(f'Skipping {y_path} because it already exists')


if __name__ == '__main__':
    # create LibriSpeech train-/dev-/test-data
    # ls_corpus = load_corpus(ls_corpus_file)
    # print(f'Creating labelled data for corpus {ls_corpus.name}')
    # create_subsets(ls_corpus, LS_TARGET_ROOT)
    # print('Done!')

    # create ReadyLingua train-/dev-/test-data
    rl_corpus = load_corpus(rl_corpus_file)
    print(f'Creating labelled data for corpus {rl_corpus.name}')
    create_subsets(rl_corpus, RL_TARGET_ROOT)
    print('Done!')
