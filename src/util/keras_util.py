import numpy as np
from keras import backend as K
from keras import callbacks
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

from util.rnn_util import decode, encode


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
        return self.next_batch()

    def __len__(self):
        return self.num_batches

    def next_batch(self):
        l = len(self.speech_segments)
        while True:
            for ndx in range(0, l, self.batch_size):
                X_data, Y_data, Y_labels = [], [], []
                for speech_segment in self.speech_segments[ndx:min(ndx + self.batch_size, l)]:
                    X_data.append(speech_segment.audio_features(self.feature_type))
                    Y_data.append(encode(speech_segment.text))
                    Y_labels.append(speech_segment.text)

                X_lengths = np.array([f.shape[0] for f in X_data])
                X = pad_sequences(X_data, maxlen=X_lengths.max(), dtype='float', padding='post', truncating='post')

                Y_lengths = np.array([len(l) for l in Y_data])
                Y = pad_sequences(Y_data, maxlen=Y_lengths.max(), dtype='int', padding='post', value=0)

                inputs = {
                    'the_input': X,
                    'the_labels': Y,
                    'input_length': X_lengths,
                    'label_length': Y_lengths,
                    'source_str': Y_labels
                }
                outputs = {'ctc': np.zeros([self.batch_size])}

                # assert X.shape == (self.batch_size, X_lengths.max(), 13)
                # assert Y.shape == (self.batch_size, Y_lengths.max())
                # assert X_lengths.shape == (self.batch_size,)
                # assert Y_lengths.shape == (self.batch_size,)

                yield inputs, outputs

            self.shuffle()


class ReportCallback(callbacks.Callback):

    def __init__(self, test_func, dev_batches: BatchGenerator, model, target_dir):
        super().__init__()
        self.test_func = test_func
        self.dev_batches = dev_batches
        self.model = model
        self.target_dir = target_dir

    def validate_epoch_end(self, verbose=0):
        for inputs, outputs in self.dev_batches:
            X = inputs['the_input']
            X_lengths = inputs['input_length']
            truths = inputs['source_str']

            y_pred = self.test_func([X])[0]
            sequences, probs = K.ctc_decode(y_pred, X_lengths, greedy=False)
            predictions = [decode(K.get_value(seq).reshape(-1)) for seq in sequences]

            for truth, pred in zip(truths, predictions):
                print(f'truth: {truth}, prediction: {pred}')
            break

    def on_epoch_end(self, epoch, logs=None):
        print(f'Validating epoch {epoch}')
        K.set_learning_phase(0)
        self.validate_epoch_end(verbose=1)
        K.set_learning_phase(1)