from abc import ABC, abstractmethod

import numpy as np
import scipy
from keras.preprocessing.image import Iterator
from keras.preprocessing.sequence import pad_sequences

from util.rnn_util import encode
from util.train_util import get_num_features


class BatchGenerator(Iterator, ABC):
    """
    Generates batches for training/validation/evaluation. Batches are created as tuples forinput and output. Each
    tuple serves as input resp. output to feed the model during training.

    Iterators are badly documented and seem to be intended for image data. Since the features (MFCC, Mel-Spectrograms
    or Power-Spectrograms) are 2D they can be considered images.

    See https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/Iterator for some information
    """

    def __init__(self, n, feature_type, batch_size, shuffle, seed):
        super().__init__(n, batch_size, shuffle, seed)
        self.num_features = get_num_features(feature_type)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        generates a batch as a tuple (inputs, outputs) with the following elements:
            inputs: array or dict for tensors 'inputs', 'labels' and 'inputs_lengths'
                X: (batch_size, T_x, num_features) --> 'inputs'
                    Input features for sequences in batch. The sequences are zero-padded to T_x whereas T_x will be the
                    length of the longest sequence. The value of num_features depends on the chosen feature type.
                Y: sparse array  --> 'labels'
                    the encoded output labels
                X_lengths: (batch_size,)  --> 'inputs_length'
                    lengths of the unpadded input sequences
                Y_lengths: (batch_size,) --> 'labels_length'
                    lengths of  the unpadded label sequences
            outputs: array or dict for tensors 'ctc' and 'decode'
                ctc: (batch_size,) zero-array
                decode: labels to decode (same as Y)

        Note that the input features will be created on-the-fly by calculating the spectrogram/MFCC per batch.
        Performance could be improved by pre-computing those features upfront and storing them in a HDF5-file.

        :param index_array: array with indices of speech segments to use for batch
        :return: (inputs, outputs) as specified above
        """
        features = self.extract_features(index_array)
        labels = self.extract_labels(index_array)

        X, X_lengths = self.make_batch_inputs(features)
        Y, Y_lengths = self.make_batch_outputs(labels)
        return [X, Y, X_lengths, Y_lengths], [np.zeros((X.shape[0],)), Y]

    @staticmethod
    def make_batch_inputs(features):
        batch_inputs = pad_sequences(features, dtype='float32', padding='post')
        batch_inputs_len = np.array([feature.shape[0] for feature in features])
        return batch_inputs, batch_inputs_len

    @staticmethod
    def make_batch_outputs(labels):
        labels_encoded = [encode(label) for label in labels]
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
    def extract_features(self, index_array):
        """
        Extract unpadded features for a batch of elements with specified indices
        :param index_array: array with indices of elements in current batch
        :return: list of unpadded features
        """
        raise NotImplementedError

    @abstractmethod
    def extract_labels(self, index_array):
        """
        Extract unpadded, unencoded labels for a batch of elements with specified indices
        :param index_array: array with indices of elements in current batch
        :return: list of textual labels
        """
        """"""
        raise NotImplementedError


class HFS5BatchGenerator(BatchGenerator):
    """
    Creates batches for by iterating over a dataset from a HDF5 file.
    """

    def __init__(self, dataset, feature_type, batch_size, shuffle=True, seed=None):
        self.inputs = dataset['inputs']
        self.labels = dataset['labels']
        self.durations = dataset['durations']
        super().__init__(len(self.inputs), feature_type, batch_size, shuffle, seed)

    def extract_features(self, index_array):
        """extract features and reshape to (T_x, num_features)"""
        return [inp.reshape((-1, self.num_features)) for inp in (self.inputs[i] for i in index_array)]

    def extract_labels(self, index_array):
        """extract labels and encode to integer"""
        return [label for label in (self.labels[i] for i in index_array)]


class OnTheFlyFeaturesIterator(BatchGenerator):
    """
    Creates batches by calculating the features on-the-fly
    """

    def __init__(self, speech_segments, feature_type, batch_size, shuffle=True, seed=None):
        self.speech_segments = np.array(speech_segments)
        super().__init__(len(self.speech_segments), feature_type, batch_size, shuffle=shuffle, seed=seed)

    def extract_features(self, index_array):
        return [seg.audio_features(self.feature_type) for seg in self.speech_segments[index_array]]

    def extract_labels(self, index_array):
        return [seg.text for seg in self.speech_segments[index_array]]
