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


def load_labelled_data(corpus_entry, root_path):
    x = y = subset_name = None

    x_pattern = re.compile(corpus_entry.id + '\.X\.(?P<subset_name>train|dev|test)\.npy')
    y_pattern = re.compile(corpus_entry.id + '\.Y\.(?P<subset_name>train|dev|test)\.npy')
    for file_name in listdir(root_path):
        x_result = x_pattern.search(file_name)
        y_result = y_pattern.search(file_name)
        if x_result:
            subset_name = x_result.group('subset_name')
            x = np.load(os.path.join(root_path, file_name))
        if y_result:
            y = np.load(os.path.join(root_path, file_name))
    return x, y, subset_name
