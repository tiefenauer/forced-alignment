import logging
import os
from os import makedirs

import numpy as np
from os.path import exists
from tqdm import tqdm

from audio_util import calculate_spectrogram
from corpus_util import load_corpus
from util import log_setup

logfile = 'create_labelled_data.log'
log_setup(filename=logfile)
log = logging.getLogger(__name__)

LS_SOURCE_ROOT = r'E:\librispeech-corpus' if os.name == 'nt' else '/media/all/D1/librispeech-corpus'
RL_SOURCE_ROOT = r'E:\readylingua-corpus' if os.name == 'nt' else '/media/all/D1/readylingua-corpus'
LS_TARGET_ROOT = r'E:\librispeech-data' if os.name == 'nt' else '/media/all/D1/librispeech-data'
RL_TARGET_ROOT = r'E:\readylingua-data' if os.name == 'nt' else '/media/all/D1/readylingua-data'

ls_corpus_file = os.path.join(LS_SOURCE_ROOT, 'librispeech.corpus')
rl_corpus_file = os.path.join(RL_SOURCE_ROOT, 'readylingua.corpus')

# for debugging only: set to a numeric value to limit the amount of processed corpus entries.
# Set to None to process all data
max_samples = None

# number of time steps in the output of the model
Ty = 1375


def create_X(corpus_entry, target_root, subset_name, overwrite=False):
    x_path = os.path.join(target_root, corpus_entry.chapter_id + '.X.' + subset_name + '.npy')
    if not exists(x_path) or overwrite:
        _, _, x = calculate_spectrogram(corpus_entry.audio_file)
        np.save(x_path, x)
    else:
        print(f'Skipping {x_path} because it already exists')


def create_Y(corpus_entry, target_root, subset_name, overwrite=False):
    return 0  # to be implemented
    Y = np.zeros(Ty, 'int16')
    for segment in corpus_entry.speech_pauses:
        if segment.segment_type == 'pause':
            Y[start:end] = 1
    return Y


def create_subsets(corpus, target_root):
    if not exists(target_root):
        makedirs(target_root)

    train_set, dev_set, test_set = corpus.train_dev_test_split()

    # limit number of samples per subset
    train_set = train_set[:max_samples] if max_samples else train_set
    dev_set = dev_set[:max_samples] if max_samples else dev_set
    test_set = test_set[:max_samples] if max_samples else test_set

    print('Creating training data...')
    for corpus_entry in tqdm(train_set, unit='corpus entry'):
        create_X(corpus_entry, target_root, 'train')
        create_Y(corpus_entry, target_root, 'train')
    print('Creating validation data...')
    for corpus_entry in tqdm(dev_set, unit='corpus entry'):
        create_X(corpus_entry, target_root, 'dev')
        create_Y(corpus_entry, target_root, 'dev')
    print('Creating test data...')
    for corpus_entry in tqdm(test_set, unit='corpus entry'):
        create_X(corpus_entry, target_root, 'test')
        create_Y(corpus_entry, target_root, 'test')


if __name__ == '__main__':
    # LibriSpeech
    ls_corpus = load_corpus(ls_corpus_file)
    print(f'Creating labelled data for corpus {ls_corpus.name}')
    create_subsets(ls_corpus, LS_TARGET_ROOT)
    print('Done!')

    # ReadyLingua
    # rl_corpus = load_corpus(rl_corpus_file)
    # print(f'Creating labelled data for corpus {ls_corpus.name}')
    # create_subsets(rl_corpus, RL_TARGET_ROOT, 'rl')
    # print('Done!')
