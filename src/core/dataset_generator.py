from abc import ABC, abstractmethod
from os import listdir
from os.path import join, splitext

import h5py
import numpy as np
import scipy
from keras.preprocessing.image import Iterator
from keras.preprocessing.sequence import pad_sequences

from util.rnn_util import encode
from util.train_util import get_num_features


class BatchGenerator(Iterator, ABC):
    """
    Generates batches for training/validation/evaluation. Batches are created as tuples of dictionaries. Each dictionary
    contains keys mapping to the data required by tensors of the model.
    """

    def __init__(self, n, feature_type, batch_size, shuffle, seed):
        super().__init__(n, batch_size, shuffle, seed)
        self.num_features = get_num_features(feature_type)

    def _get_batches_of_transformed_samples(self, index_array):
        inputs_features = self.create_input_features(index_array)
        labels_encoded = self.create_labels_encoded(index_array)

        X, X_lengths = self.make_batch_inputs(inputs_features)
        Y, Y_lengths = self.make_batch_outputs(labels_encoded)
        return [X, Y, X_lengths, Y_lengths], [np.zeros((X.shape[0],)), Y]

    def make_batch_inputs(self, inputs_features):
        batch_inputs = pad_sequences(inputs_features, dtype='float32', padding='post')
        batch_inputs_len = np.array([inp.shape[0] for inp in inputs_features])
        return batch_inputs, batch_inputs_len

    def make_batch_outputs(self, labels_encoded):
        batch_outputs_len = np.array([len(label) for label in labels_encoded])

        # the following would create labels as (padded) dense matrix, but then the receiving tensor must be dense too!
        # batch_outputs = pad_sequences(labels_encoded, dtype='int32', padding='post')

        # create labels (ground truth) as sparse matrix: more performant than dense matrix because the labels are
        # of different lengths and hence the matrix will contain a lot of zeros
        rows, cols, data = [], [], []
        for row, label in enumerate(labels_encoded):
            cols.extend(range(len(label)))
            rows.extend(len(label) * [row])
            data.extend(label)

        batch_outputs = scipy.sparse.coo_matrix((data, (rows, cols)), dtype=np.int32)
        return batch_outputs, batch_outputs_len

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        index_array.sort()
        index_array_lst = index_array.tolist()
        return self._get_batches_of_transformed_samples(index_array_lst)

    @abstractmethod
    def create_input_features(self, index_array):
        pass

    @abstractmethod
    def create_labels_encoded(self, index_array):
        pass

    @property
    def num_elements(self):
        return len(self.elements)


class OnTheFlyFeaturesIterator(BatchGenerator):

    def __init__(self, corpus_entries, feature_type, batch_size, shuffle=True, seed=None):
        speech_segs = list(seg for corpus_entry in corpus_entries for seg in corpus_entry.speech_segments_not_numeric)
        speech_segs = np.array(speech_segs)
        super().__init__(len(speech_segs), feature_type, batch_size, shuffle=shuffle, seed=seed)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        generates a batch as a tuple (inputs, outputs) with the following elements:
            inputs: array or dict for tensors 'inputs', 'labels' and 'inputs_lengths'
                X: (batch_size, T_x, num_features) --> 'inputs'
                    Input features for sequences in batch. The sequences are zero-padded to T_x whereas T_x will be the
                    length of the longest sequence. The value of num_features depends on the chosen feature type.
                Y: sparse array  --> 'labels'
                    the encoded output labels
                X_lengths: (batch_size,)  --> 'inputs_lengths'
                    lengths of the unpadded input sequences
            outputs: array or dict for tensors 'ctc' and 'decode'
                ctc: (batch_size,) zero-array
                decode: labels to decode (same as Y)

        Note that the input features will be created on-the-fly by calculating the spectrogram/MFCC per batch.
        Performance could be improved by pre-computing those features upfront and storing them in a HDF5-file.

        :param index_array: array with indices of speech segments to use for batch
        :return: (inputs, outputs) as specified above
        """
        speech_segments = self.elements[index_array]
        X_data = [seg.audio_features(self.feature_type) for seg in speech_segments]
        Y_data = [encode(seg.text) for seg in speech_segments]

        X = pad_sequences(X_data, dtype='float32', padding='post')
        X_lengths = np.array([f.shape[0] for f in X_data])

        rows, cols, data = [], [], []
        for row, label in enumerate(Y_data):
            cols.extend(range(len(label)))
            rows.extend(len(label) * [row])
            data.extend(label)

        Y = scipy.sparse.coo_matrix((data, (rows, cols)), dtype=np.int32)

        return [X, Y, X_lengths], [np.zeros((X.shape[0],)), Y]

    def create_input_features(self, index_array):
        pass

    def create_labels_encoded(self, index_array):
        pass


class HFS5BatchGenerator(BatchGenerator):
    """
    Creates batches for training/validation/evaluation by iterating over a HDF5 file.
    """

    def __init__(self, dataset, feature_type, batch_size, shuffle=True, seed=None):
        self.inputs = dataset['inputs']
        self.labels = dataset['labels']
        super().__init__(len(self.inputs), feature_type, batch_size, shuffle, seed)

    def create_input_features(self, index_array):
        return [inp.reshape((-1, self.num_features)) for inp in (self.inputs[i] for i in index_array)]

    def create_labels_encoded(self, index_array):
        return [encode(label) for label in (self.labels[i] for i in index_array)]


def generate_train_dev_test(corpus, language, feature_type, batch_size):
    h5_features = list(join(corpus.root_path, file) for file in listdir(corpus.root_path)
                       if splitext(file)[0].startswith('features')
                       and feature_type in splitext(file)[0]
                       and splitext(file)[1] == '.h5')
    default_features = f'features_{feature_type}.h5'
    feature_file = default_features if default_features in h5_features else h5_features[0] if h5_features else None
    if feature_file:
        print(f'found precomputed features: {feature_file}. Using HDF5-Features')
        f = h5py.File(feature_file, 'r')
        # hack for RL corpus: because there are no train/dev/test-subsets train/validate/test on same subset
        train_ds = f['train'][language] if corpus._name == 'LibriSpeech' else f['generic'][language]
        dev_ds = f['dev'][language] if corpus._name == 'LibriSpeech' else f['generic'][language]
        test_ds = f['test'][language] if corpus._name == 'LibriSpeech' else f['generic'][language]
        train_it = HFS5BatchGenerator(train_ds, feature_type, batch_size)
        val_it = HFS5BatchGenerator(dev_ds, feature_type, batch_size)
        test_it = HFS5BatchGenerator(test_ds, feature_type, batch_size)

    else:
        print(f'No precomputed features found. Generating features on the fly...')
        train_entries, val_entries, test_entries = corpus(languages=[language]).train_dev_test_split()
        train_it = OnTheFlyFeaturesIterator(train_entries, feature_type, batch_size)
        val_it = OnTheFlyFeaturesIterator(val_entries, feature_type, batch_size)
        test_it = OnTheFlyFeaturesIterator(test_entries, feature_type, batch_size)

    return train_it, val_it, test_it
