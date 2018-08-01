"""
This file can be used if a corpus was created on the cluster (Unix) and downloaded locally (Windows) or vice versa.
Since the paths to the audio files are absolute, this script will fix them.
"""

# set corpus ID here
from os.path import basename, join

from tqdm import tqdm

from constants import LS_CORPUS_ROOT, RL_CORPUS_ROOT
from util.corpus_util import get_corpus, save_corpus

corpus_id = 'ls'

if __name__ == '__main__':
    corpus = get_corpus(corpus_id)
    corpus.root_path = LS_CORPUS_ROOT if corpus_id == 'ls' else RL_CORPUS_ROOT
    progress = tqdm(corpus, unit=' corpus entries')
    for corpus_entry in progress:
        old_path = corpus_entry.audio_file
        new_path = join(corpus.root_path, basename(corpus_entry.audio_file))
        corpus_entry.audio_file = new_path
        progress.set_description(f'{old_path} changed to {new_path}')
    save_corpus(corpus, corpus.root_path)
