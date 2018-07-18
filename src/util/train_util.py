import os
import sys
from datetime import datetime

from constants import POC_PROFILES
from util.corpus_util import load_corpus
from util.log_util import create_args_str, print_to_file_and_console


def get_num_features(feature_type):
    if feature_type == 'pow':
        return 161
    elif feature_type == 'mel':
        return 40
    elif feature_type == 'mfcc':
        return 13
    print(f'error: unknown feature type: {feature_type}', file=sys.stderr)
    exit(1)


def get_poc(args):
    print(create_args_str(args))
    if args.poc:
        print(f'applying profile for PoC#{args.poc}')
        poc = POC_PROFILES['poc_' + args.poc]
        for key, value in poc.items():
            setattr(args, key, value)
    print(create_args_str(args))
    return args


def get_target_dir(rnn_type, args):
    target_dir = os.path.join(args.target_root, datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + rnn_type)
    target_dir += '_'.join(['_poc_' + args.poc if hasattr(args, 'poc') else '',
                            args.architecture.upper() if hasattr(args, 'architecture') else '',
                            args.language,
                            args.feature_type,
                            'synthesized' if hasattr(args, 'synthesize') and args.synthesize else ''])

    print(f'Results will be written to: {target_dir}')
    log_file_path = os.path.join(target_dir, 'train.log')
    print_to_file_and_console(log_file_path)  # comment out to only log to console
    return target_dir


def get_corpus(args):
    ls_corpus_root = os.path.join(args.target_root, 'librispeech-corpus')
    rl_corpus_root = os.path.join(args.target_root, 'readylingua-corpus')
    if args.corpus == 'rl':
        corpus = load_corpus(rl_corpus_root)
    elif args.corpus == 'ls':
        corpus = load_corpus(ls_corpus_root)
    corpus = corpus(languages=args.language)
    corpus.summary()
    return corpus