import argparse
import os

import numpy as np
from keras import Model
from keras import backend as K
from keras.activations import relu
from keras.initializers import random_normal
from keras.layers import Dense, Dropout, Input, TimeDistributed, Bidirectional, LSTM, Lambda
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import get_custom_objects

from constants import TRAIN_TARGET_ROOT
from train_rnn import MAX_EPOCHS
from util.rnn_util import encode
from util.train_util import get_num_features, get_corpus

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

parser = argparse.ArgumentParser(
    description="""Train Bi-directionalRNN with CTC cost function for speech recognition""")
parser.add_argument('-c', '--corpus', type=str, choices=['rl', 'ls'], default='ls',
                    help='corpus on which to train the RNN (rl=ReadyLingua, ls=LibriSpeech')
parser.add_argument('-l', '--language', type=str, default='en',
                    help='language on which to train the RNN')
parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=5,
                    help=f'(optional) number of speech segments to include in one batch (default: 5)')
parser.add_argument('-f', '--feature_type', type=str, nargs='?', choices=['mfcc', 'mel', 'pow'], default='mfcc',
                    help=f'(optional) features to use for training (default: mfcc)')
parser.add_argument('-t', '--target_root', type=str, nargs='?', default=TRAIN_TARGET_ROOT,
                    help=f'(optional) root directory where results will be written to (default: {TRAIN_TARGET_ROOT})')
parser.add_argument('-e', '--num_epochs', type=int, nargs='?', default=MAX_EPOCHS,
                    help=f'(optional) number of epochs to train the model (default: {MAX_EPOCHS})')
args = parser.parse_args()


def main():
    print('loading train-/dev-/test-set')
    corpus = get_corpus(args)
    train_set, dev_set, test_set = corpus.train_dev_test_split()
    print(f'train/dev/test: {len(train_set)}/{len(dev_set)}/{len(test_set)} '
          f'({100*len(train_set)//len(corpus)}/{100*len(dev_set)//len(corpus)}/{100*len(test_set)//len(corpus)}%)')

    # create model
    num_features = get_num_features('mfcc')
    model = create_model(num_features)

    # train model
    train_model(model, train_set, dev_set, test_set)


def create_model(num_features, fc_size=2048, rnn_size=512, dropout=[0.1, 0.1, 0.1], output_dim=28):
    """ DeepSpeech 1 Implementation with Dropout

    Architecture:
        Input MFCC TIMEx26
        3 Fully Connected using Clipped Relu activation function
        3 Dropout layers between each FC
        1 BiDirectional LSTM
        1 Dropout applied to BLSTM
        1 Dropout applied to FC dense
        1 Fully connected Softmax

    Details:
        - Uses MFCC's rather paper's 80 linear spaced log filterbanks
        - Uses LSTM's rather than SimpleRNN
        - No translation of raw audio by 5ms
        - No stride the RNN

    Reference:
        https://arxiv.org/abs/1412.5567
    """
    get_custom_objects().update({"clipped_relu": clipped_relu})

    input_data = Input(shape=(None, num_features), name='the_input')
    init = random_normal(stddev=0.046875)

    x = TimeDistributed(
        Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(input_data)

    # Layers 1-3: FC
    x = TimeDistributed(Dropout(dropout[0]))(x)
    x = TimeDistributed(
        Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)
    x = TimeDistributed(Dropout(dropout[0]))(x)
    x = TimeDistributed(
        Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation=clipped_relu))(x)
    x = TimeDistributed(Dropout(dropout[0]))(x)

    # Layer 4 BiDirectional RNN
    x = Bidirectional(LSTM(rnn_size, return_sequences=True, activation=clipped_relu, dropout=dropout[1],
                           kernel_initializer='he_normal', name='birnn'), merge_mode='sum')(x)

    # Layer 5+6 Time Dist Dense Layer & Softmax
    # x = TimeDistributed(Dense(fc_size, activation=clipped_relu, kernel_initializer=init, bias_initializer=init))(x)
    x = TimeDistributed(Dropout(dropout[2]))(x)
    y_pred = TimeDistributed(
        Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax"),
        name="out")(x)

    # Change shape
    labels = Input(name='the_labels', shape=[None, ], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.summary()
    return model


def train_model(model, train_set, dev_set, test_set):
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    model.compile(optimizer=opt, loss=ctc)

    model.fit_generator(generator=next_batch(train_set),
                        steps_per_epoch=len(train_set) // args.batch_size,
                        epochs=1,
                        # callbacks=cb_list,
                        # validation_data=next_batch(dev_set),
                        # validation_steps=len(dev_set) // args.batch_size,
                        initial_epoch=0,
                        verbose=1,
                        class_weight=None,
                        max_q_size=10,
                        workers=1,
                        pickle_safe=False
                        )


def next_batch(corpus_entries):
    speech_segments = list(seg for corpus_entry in corpus_entries for seg in corpus_entry.speech_segments_not_numeric)
    l = len(speech_segments)
    for ndx in range(0, l, args.batch_size):
        X_data, Y_labels = [], []
        for speech_segment in speech_segments[ndx:min(ndx + args.batch_size, l)]:
            X_data.append(speech_segment.mfcc())
            Y_labels.append(encode(speech_segment.text))

        X_lengths = np.array([f.shape[0] for f in X_data])
        X = pad_sequences(X_data, maxlen=X_lengths.max(), dtype='float', padding='post', truncating='post')

        Y_lengths = np.array([len(l) for l in Y_labels])
        Y = pad_sequences(Y_labels, maxlen=Y_lengths.max(), dtype='int', padding='post', value=0)

        inputs = {'the_input': X, 'the_labels': Y, 'input_length': X_lengths, 'label_length': Y_lengths}
        outputs = {'ctc': np.zeros([args.batch_size])}

        assert X.shape == (args.batch_size, X_lengths.max(), 13)
        assert Y.shape == (args.batch_size, Y_lengths.max())
        assert X_lengths.shape == (args.batch_size,)
        assert Y_lengths.shape == (args.batch_size,)

        yield (inputs, outputs)


def clipped_relu(x):
    return relu(x, max_value=20)


def ctc(y_true, y_pred):
    return y_pred


# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # hack for load_model

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    # print("CTC lambda inputs / shape")
    # print("y_pred:",y_pred.shape)  # (?, 778, 30)
    # print("labels:",labels.shape)  # (?, 80)
    # print("input_length:",input_length.shape)  # (?, 1)
    # print("label_length:",label_length.shape)  # (?, 1)

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


if __name__ == '__main__':
    main()
