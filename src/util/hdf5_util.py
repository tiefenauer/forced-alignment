from os import remove
from os.path import join

import h5py
import numpy as np

from constants import ROOT_DIR


def combine_subsets(subsets):
    tmp_file = join(ROOT_DIR, 'tmp', 'tmp.h5')
    f_tmp = h5py.File(tmp_file, 'w')
    tmp_group = f_tmp.create_group('tmp_group')
    inputs = tmp_group.create_dataset('inputs', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.float32))
    labels = tmp_group.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.unicode))
    duration = tmp_group.create_dataset('duration', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.float32))
    for subset in subsets:
        inps = subset['inputs']
        lbls = subset['labels']
        durs = subset['durations']

        inputs.resize(inputs.shape[0] + inps.shape[0], axis=0)
        for inp in inps:
            inputs[inputs.shape[0] - 1] = inp

        labels.resize(labels.shape[0] + lbls.shape[0], axis=0)
        labels[labels.shape[0] - 1] = lbls

        duration.resize(duration.shape[0] + durs.shape[0], axis=0)
        duration[duration.shape[0] - 1] = durs
    remove(f_tmp)
    return tmp_group
