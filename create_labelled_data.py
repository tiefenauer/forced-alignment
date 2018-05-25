import argparse
import logging
import os

import numpy as np
from os.path import exists
from tqdm import tqdm

from audio_util import log_specgram
from corpus_util import load_corpus, save_corpus
from log_util import log_setup, create_args_str

logfile = 'create_labelled_data.log'
log_setup(filename=logfile)
log = logging.getLogger(__name__)
# -------------------------------------------------------------
# Constants, defaults and env-vars
# -------------------------------------------------------------
DEFAULT_CORPUS_ROOT = r'E:\\' if os.name == 'nt' else '/media/all/D1/'
RL_CORPUS_ROOT = os.path.join(DEFAULT_CORPUS_ROOT, 'readylingua-corpus')
LS_CORPUS_ROOT = os.path.join(DEFAULT_CORPUS_ROOT, 'ibrispeech-corpus')

# -------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description='Create labelled train-, dev- and test-data (X and Y) for all corpora')
parser.add_argument('-f', '--file', help='Dummy argument for Jupyter Notebook compatibility')
parser.add_argument('corpus', nargs='?', type=str, choices=['rl', 'ls'],
                    help='(optional) select which corpus to process (rl=ReadyLingua, ls=LibriSpeech). '
                         'Default=None (all)')
parser.add_argument('-d', '--corpus_root', default=DEFAULT_CORPUS_ROOT,
                    help=f'(optional) root directory where the corpora are stored. Default={DEFAULT_CORPUS_ROOT}')
parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                    help='(optional) overwrite existing data if already present. Default=False)')
parser.add_argument('-m', '--max_samples', type=int, default=None,
                    help='(optional) maximum number of samples to proces per subset. Default=None (all)')
parser.add_argument('-X', '--no_spectrograms', default=False, action='store_true',
                    help='(optional) don\'t create spectrograms (X). Default=False')
parser.add_argument('-Y', '--no_labels', default=False, action='store_true',
                    help='(optional) don\'t create labels (Y). Default=False')
parser.add_argument('-ty', '--Ty', type=int, default=1375, help='Number of steps in the RNN output layer (T_y)')
args = parser.parse_args()

# -------------------------------------------------------------
# Other values
# -------------------------------------------------------------
T_y = args.Ty


def main():
    print(create_args_str(args))

    # create LibriSpeech train-/dev-/test-data
    if not args.corpus or args.corpus == 'ls':
        print(f'Processing files from {LS_CORPUS_ROOT}')
        create_X_Y(LS_CORPUS_ROOT, 'librispeech.corpus', args.no_spectrograms, args.no_labels, args.max_samples)
        print('Done!')

    # create ReadyLingua train-/dev-/test-data
    if not args.corpus or args.corpus == 'rl':
        print(f'Processing files from {RL_CORPUS_ROOT}')
        create_X_Y(RL_CORPUS_ROOT, 'readylingua.corpus', args.no_spectrograms, args.no_labels, args.max_samples)
        print('Done!')


def create_X_Y(corpus_root, corpus_file, no_spectrograms=False, no_labels=False, max_samples=None):
    corpus_path = os.path.join(corpus_root, corpus_file)
    corpus = load_corpus(corpus_path)
    for corpus_entry in tqdm(corpus[:max_samples], unit=' corpus entries'):
        if not no_spectrograms:
            corpus_entry.x_path = create_x(corpus_entry, corpus_root)
        if not no_labels:
            corpus_entry.y_path = create_y(corpus_entry, corpus_root)
    save_corpus(corpus, corpus_path)


def create_x(corpus_entry, corpus_root):
    x_path = os.path.join(corpus_root, corpus_entry.id + '.X.npy')
    if not exists(x_path) or args.overwrite:
        rate, audio = corpus_entry.audio
        freqs, times, spec = log_specgram(audio, rate)
        np.save(x_path, (freqs, times, spec))
    else:
        print(f'Skipping {x_path} because it already exists')
    return x_path


def create_y(corpus_entry, corpus_root):
    y_path = os.path.join(corpus_root, corpus_entry.id + '.Y.npy')
    if not exists(y_path) or args.overwrite:
        duration = float(corpus_entry.media_info['duration'])
        sample_rate = float(corpus_entry.media_info['sample_rate'])
        n_frames = int(duration * sample_rate)

        # initialize label vector with zeroes (=speech)
        y = np.zeros((1, T_y), 'int16')

        # set fraction of label vector to one for each speech segment
        for pause_segments in corpus_entry.pause_segments:
            start = round(pause_segments.start_frame * T_y / n_frames)
            end = round(pause_segments.end_frame * T_y / n_frames)
            y[:, start:end] = 1
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

    return y_path


if __name__ == '__main__':
    main()
