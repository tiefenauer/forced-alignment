import argparse

import random

from util.corpus_util import get_corpus
from util.e2e_util import create_demo_from_corpus_entry

parser = argparse.ArgumentParser(description="Create E2E Demo either from a combination of corpus/id or from raw data")
parser.add_argument('-c', '--corpus', type=str, nargs='?', choices=['rl', 'ls'],
                    help='(optional) corpus to use. If this is set, all other arguments are disregarded')
parser.add_argument('-id', type=str, nargs='?',
                    help='(optional) ID of corpus entry to use. If not set a random entry will be chosen.')
parser.add_argument('-a', '--audio', type=str, nargs='?',
                    help='(optional) path to audio file')
parser.add_argument('-t', '--transcription', type=str, nargs='?',
                    help='(optional) path to transcription file')
args = parser.parse_args()

if __name__ == '__main__':
    if args.corpus:
        corpus = get_corpus(args.corpus)
        corpus_entry = corpus[args.id] if args.id is not None else random.choice(corpus)
        create_demo_from_corpus_entry(corpus_entry)
    else:
        raise ValueError('demo creation from raw data not implemented yet')
