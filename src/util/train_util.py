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
    target_dir += '_'.join(['_poc_' + args.poc if hasattr(args, 'poc') else '',
                            args.architecture.upper() if hasattr(args, 'architecture') else '',
                            args.language,
                            args.feature_type,
                            'synthesized' if hasattr(args, 'synthesize') and args.synthesize else ''])
    return target_dir
