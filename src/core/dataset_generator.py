import numpy as np
import scipy
from keras.preprocessing.image import Iterator
from keras.preprocessing.sequence import pad_sequences

from util.rnn_util import encode


class BatchGenerator(Iterator):
    """
    Generates batches for training/validation/evaluation. Batches are created as tuples of dictionaries. Each dictionary
    contains keys mapping to the data required by tensors of the model.
    """

    def __init__(self, corpus_entries, feature_type, batch_size, shuffle=True, seed=None):
        speech_segs = list(seg for corpus_entry in corpus_entries for seg in corpus_entry.speech_segments_not_numeric)
        super().__init__(len(speech_segs), batch_size, shuffle, seed)
        self.speech_segments = np.array(speech_segs)
        self.feature_type = feature_type

    @property
    def num_elements(self):
        return len(self.speech_segments)

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
        X_data = [seg.audio_features(self.feature_type) for seg in self.speech_segments[index_array]]
        X = pad_sequences(X_data, dtype='float32', padding='post')
        X_lengths = np.array([f.shape[0] for f in X_data])

        Y_data = [encode(seg.text) for seg in self.speech_segments[index_array]]
        rows, cols, data = [], [], []
        for row, label in enumerate(Y_data):
            cols.extend(range(len(label)))
            rows.extend(len(label) * [row])
            data.extend(label)

        Y = scipy.sparse.coo_matrix((data, (rows, cols)), dtype='int32')

        return [X, Y, X_lengths], [np.zeros((X.shape[0],)), Y]
