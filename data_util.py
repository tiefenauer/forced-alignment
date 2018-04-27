import os

import numpy as np


def save_spectrogram(spectrogram, target_root, audio_name, subset_name):
    file_path = os.path.join(target_root, audio_name + '.' + subset_name)
    np.save(file_path, spectrogram)


def save_subset(X, Y, target_root, infix, subset_name):
    X_path = os.path.join(target_root, create_subset_file_name('X', subset_name, infix))
    Y_path = os.path.join(target_root, create_subset_file_name('Y', subset_name, infix))
    np.save(X_path, X)
    np.save(Y_path, Y)
    print(f'\t saved {len(X)} X-{subset_name} samples to {X_path}')
    print(f'\t saved {len(Y)} Y-{subset_name} samples to {Y_path}')


def load_subset(subset_name, infix, root_path):
    X_fn = create_subset_file_name('X', subset_name, infix)
    Y_fn = create_subset_file_name('Y', subset_name, infix)
    X_path = os.path.join(root_path, X_fn)
    Y_path = os.path.join(root_path, Y_fn)

    X = np.load(X_path)
    Y = np.load(Y_path)
    return X, Y


def create_subset_file_name(X_or_Y, subset_name, infix):
    return '_'.join([X_or_Y, subset_name]) + '.' + infix
