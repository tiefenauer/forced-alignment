import os
import sys
from datetime import datetime

from constants import POC_PROFILES
from util.log_util import create_args_str, redirect_to_file


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
    if hasattr(args, 'poc'):
        target_dir += '_poc'
    if hasattr(args, 'architecture'):
        target_dir += '_' + args.architecture.upper()
    target_dir += '_' + args.corpus
    target_dir += '_' + args.language
    target_dir += '_' + args.feature_type
    if hasattr(args, 'synthesize'):
        target_dir += '_synthesized'
    return target_dir
