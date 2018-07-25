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
    progress = tqdm(list(subset_segments), unit='speech segments')
    data = []
    for subset, speech_segment in progress:
        features = speech_segment.audio_features(feature_type)
        labels = speech_segment.text
        duration = speech_segment.audio_length
        data.append({'subset': subset, 'input': features, 'label': labels, 'duration': duration})
        desc = f'subset: {subset}, t_x: {len(features)}, t_y: {len(labels)}, duration: {timedelta(seconds=duration)}'
        progress.set_description(desc)

    print(f'created features for {len(data)} speech segments. Saving to {target_file}...')
    with h5py.File(target_file) as f:
        data.sort(key=lambda item: item['subset'])
        for subset, items in tqdm(groupby(data, key=lambda item: item['subset']), unit='datasets'):
            items = list(items)
            group = f.create_group(subset) if subset not in f else f[subset]

            dt_inputs = h5py.special_dtype(vlen=np.float64)
            inputs = group.create_dataset(name='inputs', shape=(0,), maxshape=(None,), dtype=dt_inputs)
            for input in [item['input'] for item in items]:
                inputs.resize(inputs.shape[0] + 1, axis=0)
                inputs[inputs.shape[0] - 1] = input.flatten().astype(np.float32)

            dt_labels = h5py.special_dtype(vlen=np.unicode)
            labels = group.create_dataset(name='labels', shape=(0,), maxshape=(None,), dtype=dt_labels)
            for label in [item['label'] for item in items]:
                labels.resize(labels.shape[0] + 1, axis=0)
                labels[labels.shape[0] - 1] = label.encode('utf8')

            durations = group.create_dataset(name='durations', shape=(0,), maxshape=(None,))
            for duration in [item['duration'] for item in items]:
                durations.resize(durations.shape[0] + 1, axis=0)
                durations[durations.shape[0] - 1] = duration

        f.flush()
    print(f'...done! {len(data)} datasets saved in {target_file}')


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
