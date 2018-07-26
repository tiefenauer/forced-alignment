import argparse
from datetime import timedelta
from itertools import takewhile
from os import makedirs, remove
from os.path import exists, join, dirname

import h5py
import numpy as np
from tqdm import tqdm

from util.corpus_util import get_corpus

parser = argparse.ArgumentParser(description="""Precompute audio features and labels for speech segments""")
parser.add_argument('-c', '--corpus', nargs='?', type=str, choices=['rl', 'ls'], default='rl',
                    help='(optional) corpus to create features for')
parser.add_argument('-f', '--feature_type', nargs='?', type=str, choices=['mfcc', 'mel', 'pow'], default=None,
                    help='(optional) feature type to precompute (default: all)')
parser.add_argument('-t', '--target_file', nargs='?', type=str, default=None,
                    help='(optional) target directory to save results (default: corpus directory)')
parser.add_argument('-l', '--limit', nargs='?', type=int, default=None,
                    help='(optional) maximum number of speechs egments to process')
args = parser.parse_args()


def generate_speech_segments(corpus):
    """
    create a generateor of (subset, segment)-tuples from a corpus whereas subset is either train|dev|test and segment
    is a speech segment that does not contain numbers in its transcrupt
    :param corpus: a corpus
    :return: generator as specified above
    """
    return ((corpus_entry.subset, seg) for corpus_entry in corpus for seg in corpus_entry.speech_segments_not_numeric)


def precompute_features(corpus, feature_type, target_file, limit=None):
    # get number of elements and delete it right afterwards to prevent memory error
    subset_segments = list(generate_speech_segments(corpus))
    num_segments = len(subset_segments)
    del subset_segments

    if limit:
        subset_segments = list(takewhile(lambda x: x[0] < limit, generate_speech_segments(corpus)))
    else:
        subset_segments = generate_speech_segments(corpus)

    with h5py.File(target_file) as f:
        progress = tqdm(enumerate(subset_segments), total=num_segments, unit='speech segments')
        for i, (subset, speech_segment) in progress:
            inp = speech_segment.audio_features(feature_type)
            lbl = speech_segment.text
            dur = speech_segment.audio_length
            set_name = 'train' if subset.startswith('train') else 'dev' if subset.startswith('dev') else 'test'
            desc = f'set: {set_name}, subset: {subset}, t_x: {len(inp)}, t_y: {len(lbl)}, duration: {timedelta(seconds=dur)}'
            progress.set_description(desc)

            if set_name not in f:
                group = f.create_group(set_name)
                group.attrs['subset'] = subset
                dt_inputs = h5py.special_dtype(vlen=np.float32)
                group.create_dataset(name='inputs', shape=(0,), maxshape=(None,), dtype=dt_inputs)

                dt_labels = h5py.special_dtype(vlen=np.unicode)
                group.create_dataset(name='labels', shape=(0,), maxshape=(None,), dtype=dt_labels)

                group.create_dataset(name='durations', shape=(0,), maxshape=(None,))

            inputs = f[set_name]['inputs']
            labels = f[set_name]['labels']
            durations = f[set_name]['durations']

            inputs.resize(inputs.shape[0] + 1, axis=0)
            inputs[inputs.shape[0] - 1] = inp.flatten().astype(np.float32)

            labels.resize(labels.shape[0] + 1, axis=0)
            labels[labels.shape[0] - 1] = lbl.encode('utf8')

            durations.resize(durations.shape[0] + 1, axis=0)
            durations[durations.shape[0] - 1] = dur

            if i % 256 == 0:
                f.flush()

        f.flush()
    print(f'...done! {i} datasets saved in {target_file}')


def get_target_file(corpus, feature_type, target_file):
    prefix = target_file if target_file else 'features'
    return join(corpus.root_path, f'{prefix}_{feature_type}.h5')


def check_target_files(corpus, feature_types, file):
    check_files = [(feature, get_target_file(corpus, feature, file)) for feature in feature_types]
    feature_files = []
    for feature, file in check_files:
        if exists(file):
            override = input(f'target file {file} already exists. Overwrite? [Y/n]')
            if override.lower() in ['', 'y']:
                remove(file)
                feature_files.append((feature, file))
        else:
            feature_files.append((feature, file))
    return feature_files


if __name__ == '__main__':
    corpus = get_corpus(args.corpus)
    feature_types = [args.feature_type] if args.feature_type else ['mfcc', 'mel', 'pow']
    target_files = check_target_files(corpus, feature_types, args.target_file)
    for feature_type, target_file in target_files:
        if not exists(dirname(target_file)):
            makedirs(dirname(target_file))
        print(f'precomputing {feature_type} features. Results will be written to {target_file}')
        precompute_features(corpus, feature_type, target_file, args.limit)
