"""
This file can be used if a corpus was created on the cluster (Unix) and downloaded locally (Windows) or vice versa.
Since the paths to the audio files are absolute, this script will fix them.
"""
import argparse

from constants import RL_CORPUS_DIR, LS_CORPUS_DIR
from util.log_util import create_args_str

parser = argparse.ArgumentParser(
    description="""change paths of corpus entries to new root""")
parser.add_argument('corpus_id', type=str, choices=['rl', 'ls'], help='corpus to use (rl=ReadyLingua, ls=LibriSpeech')
parser.add_argument('new_root', type=str, help=f'new root path of')
args = parser.parse_args()

from os.path import basename, join

from tqdm import tqdm

from util.corpus_util import get_corpus, save_corpus

if __name__ == '__main__':
    print(create_args_str(args))
    corpus_id, new_root = args.corpus_id, args.new_root
    corpus = get_corpus(corpus_id, corpus_root=new_root)
    corpus_dir = RL_CORPUS_DIR if corpus_id == 'rl' else LS_CORPUS_DIR
    new_root = join(args.new_root, corpus_dir)
    print(f'changing corpus root from {corpus.root_path} to {args.new_root}')
    corpus.root_path = args.new_root

    progress = tqdm(corpus, unit=' corpus entries')
    for corpus_entry in progress:
        old_path = corpus_entry.audio_file
        new_path = join(corpus.root_path, basename(corpus_entry.audio_file))
        corpus_entry.audio_file = new_path
        progress.set_description(f'{old_path} changed to {new_path}')
    save_corpus(corpus, corpus.root_path)
