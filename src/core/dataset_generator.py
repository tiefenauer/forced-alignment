from abc import ABC, abstractmethod

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
        inputs = self.create_input_features(index_array)
        labels = self.create_labels_encoded(index_array)

        X, X_lengths = self.make_batch_input(inputs)
        Y = self.make_batch_output(labels)
        return [X, Y, X_lengths], [np.zeros((X.shape[0],)), Y]

    def make_batch_input(self, inputs_features):
        batch_inputs = pad_sequences(inputs_features, dtype='float32', padding='post')
        batch_inputs_len = np.array([inp.shape[0] for inp in inputs_features])
        return batch_inputs, batch_inputs_len

    def make_batch_output(self, labels_encoded):
        rows, cols, data = [], [], []
        for row, label in enumerate(labels_encoded):
            cols.extend(range(len(label)))
            rows.extend(len(label) * [row])
            data.extend(label)

        return scipy.sparse.coo_matrix((data, (rows, cols)), dtype='int32')

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

        Y = scipy.sparse.coo_matrix((data, (rows, cols)), dtype='int32')

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
