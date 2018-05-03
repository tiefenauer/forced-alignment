import os
import re
from glob import glob
from os import listdir

import numpy as np


def load_subset(subset_name, root_path):
    glob_pattern = os.path.join(root_path, '*' + subset_name + '.npy')
    file_pattern = re.compile('(?P<entry_id>\d*)\.X\.train\.npy')
    for filename in glob(glob_pattern):
        result = re.search(file_pattern, filename)
        if result:
            entry_id = result.group('entry_id')
            x_path = glob_pattern.replace('*' + subset_name + '.npy', entry_id + '.X.' + subset_name + '.npy')
            y_path = glob_pattern.replace('*' + subset_name + '.npy', entry_id + '.Y.' + subset_name + '.npy')
            x = np.load(x_path)
            y = np.load(y_path)
            yield x, y


def save_x(freqs, times, spec, x_path):
    np.save(x_path, (freqs, times, spec))


def load_x(corpus_entry, root_path):
    file_path, subset_name = find_file_by_corpus_entry_id(corpus_entry.id, 'X', root_path)
    if file_path:
        (freqs, times, spec) = np.load(file_path)
        return freqs, times, spec


def load_y(corpus_entry, root_path):
    file_path, subset_name = find_file_by_corpus_entry_id(corpus_entry.id, 'Y', root_path)
    if file_path:
        return np.load(file_path)


def find_file_by_corpus_entry_id(corpus_entry_id, X_or_Y, root_path):
    pattern = re.compile(corpus_entry_id + '\.' + X_or_Y + '\.(?P<subset_name>train|dev|test)\.npy')
    for file_name in listdir(root_path):
        result = re.search(pattern, file_name)
        if result:
            file_path = os.path.join(root_path, file_name)
            subset_name = result.group('subset_name')
            return file_path, subset_name
