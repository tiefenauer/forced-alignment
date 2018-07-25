import argparse
from datetime import timedelta
from itertools import groupby
from os import makedirs, remove
from os.path import exists, join, dirname

import h5py
import numpy as np
from tqdm import tqdm

from util.corpus_util import get_corpus

parser = argparse.ArgumentParser(description="""Precompute audio features and labels""")
parser.add_argument('-c', '--corpus', nargs='?', type=str, choices=['rl', 'ls'], default='rl',
                    help='(optional) corpus to create features for')
parser.add_argument('-f', '--feature_type', nargs='?', type=str, choices=['mfcc', 'mel', 'pow'], default=None,
                    help='(optional) feature type to precompute (default: all)')
parser.add_argument('-t', '--target_file', nargs='?', type=str,
                    help='(optional) target directory to save results (default: corpus directory)')
args = parser.parse_args()


def precompute_features(corpus, feature_type, target_file):
    subset_segments = ((corpus_entry.subset, seg) for corpus_entry in corpus for seg in
                       corpus_entry.speech_segments_not_numeric)
    progress = tqdm(enumerate(list(subset_segments)), unit='speech segments')

    with h5py.File(target_file) as f:
        for i, (subset, speech_segment) in progress:
            inp = speech_segment.audio_features(feature_type)
            lbl = speech_segment.text
            dur = speech_segment.audio_length
            desc = f'subset: {subset}, t_x: {len(inp)}, t_y: {len(lbl)}, duration: {timedelta(seconds=dur)}'
            progress.set_description(desc)

            if subset not in f:
                group = f.create_group(subset)
                dt_inputs = h5py.special_dtype(vlen=np.float64)
                group.create_dataset(name='inputs', shape=(0,), maxshape=(None,), dtype=dt_inputs)

                dt_labels = h5py.special_dtype(vlen=np.unicode)
                group.create_dataset(name='labels', shape=(0,), maxshape=(None,), dtype=dt_labels)

                group.create_dataset(name='durations', shape=(0,), maxshape=(None,))

            inputs = f[subset]['inputs']
            labels = f[subset]['labels']
            durations = f[subset]['durations']

            inputs.resize(inputs.shape[0] + 1, axis=0)
            inputs[inputs.shape[0] - 1] = inp.flatten().astype(np.float32)

            labels.resize(labels.shape[0] + 1, axis=0)
            labels[labels.shape[0] - 1] = lbl.encode('utf8')

            durations.resize(durations.shape[0] + 1, axis=0)
            durations[durations.shape[0] - 1] = dur

            if i % 128 == 0:
                f.flush()

        f.flush()
    print(f'...done! {i} datasets saved in {target_file}')


if __name__ == '__main__':
    corpus = get_corpus(args.corpus)
    feature_types = [args.feature_type] if args.feature_type else ['mfcc', 'mel', 'pow']
    for feature_type in feature_types:
        target_file = f'{args.target_file}.{feature_type}' if args.target_file else join(corpus.root_path,
                                                                                         f'features_{feature_type}.h5')
        if not exists(dirname(target_file)):
            makedirs(dirname(target_file))

        # if exists(target_file):
        #     override = input(f'target file {target_file} already exists. Overwrite? [Y/n]')
        #     if override.lower() in ['', 'y']:
        #         remove(target_file)
        print(f'precomputing {feature_type} features. Results will be written to {target_file}')
        precompute_features(corpus, feature_type, target_file)
