import os
import re
import numpy as np
from glob import glob


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


def create_subset_file_name(X_or_Y, subset_name, infix):
    return '_'.join([X_or_Y, subset_name]) + '.' + infix
