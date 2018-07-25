import numpy as np
import scipy
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

from util.rnn_util import encode


class BatchGenerator(object):
    """
    Generates batches for training/validation/evaluation. Batches are created as tuples of dictionaries. Each dictionary
    contains keys mapping to the data required by tensors of the model.
    """

    def __init__(self, corpus_entries, feature_type, batch_size, steps=None):
        self.speech_segments = list(
            seg for corpus_entry in corpus_entries for seg in corpus_entry.speech_segments_not_numeric)
        self.feature_type = feature_type
        self.batch_size = batch_size
        self.num_batches = steps if steps else len(self.speech_segments) // self.batch_size

    def shuffle(self):
        self.speech_segments = shuffle(self.speech_segments)

    def __iter__(self):
        return self.generate_batches()

    def __len__(self):
        return self.num_batches

    def generate_batches(self):
        return SpeechSegmentIterator(self.speech_segments, self.feature_type, self.batch_size)

        # l = len(self.speech_segments)
        # while True:
        #     for ndx in range(0, l, self.batch_size):
        #         X_data, Y_data, Y_labels = [], [], []
        #         for speech_segment in self.speech_segments[ndx:min(ndx + self.batch_size, l)]:
        #             X_data.append(speech_segment.audio_features(self.feature_type))
        #             Y_data.append(encode(speech_segment.text))
        #             Y_labels.append(speech_segment.text)
        #
        #         X_lengths = np.array([f.shape[0] for f in X_data])
        #         X = pad_sequences(X_data, dtype='float32', padding='post')
        #
        #         Y = pad_sequences(Y_data, dtype='int', padding='post', value=0)
        #
        #         inputs = {
        #             'inputs': X,
        #             'inputs_length': X_lengths
        #             'labels': Y
        #         }
        #         outputs = {'ctc': np.zeros([self.batch_size])}
        #
        #         # assert X.shape == (self.batch_size, X_lengths.max(), 13)
        #         # assert Y.shape == (self.batch_size, Y_lengths.max())
        #         # assert X_lengths.shape == (self.batch_size,)
        #         # assert Y_lengths.shape == (self.batch_size,)
        #
        #         yield inputs, outputs
        #
        #     self.shuffle()


from keras.preprocessing.image import Iterator


class SpeechSegmentIterator(Iterator):

    def __init__(self, speech_segments, feature_type, batch_size, shuffle=True, seed=None):
        super().__init__(len(speech_segments), batch_size, shuffle, seed)
        self.speech_segments = np.array(speech_segments)
        self.feature_type = feature_type

    def _get_batches_of_transformed_samples(self, index_array):
        X_data, X_lengths, Y_data, Y_labels = [], [], [], []
        for speech_segment in self.speech_segments[index_array]:
            X_data.append(speech_segment.audio_features(self.feature_type))
            Y_data.append(encode(speech_segment.text))
            Y_labels.append(speech_segment.text)

        X = pad_sequences(X_data, dtype='float32', padding='post')
        X_lengths = np.array([f.shape[0] for f in X_data])
        Y = pad_sequences(Y_data, dtype='int', padding='post', value=0)

        rows, cols, data = [], [], []
        for row, label in enumerate(Y_data):
            cols.extend(range(len(label)))
            rows.extend(len(label) * [row])
            data.extend(label)

        batch_labels = scipy.sparse.coo_matrix((data, (rows, cols)), dtype='int32')

        return [X, batch_labels, X_lengths], [np.zeros((X.shape[0],)), batch_labels]
