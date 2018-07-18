# https://github.com/robmsmt/KerasDeepSpeech
import argparse
import os

import tensorflow as tf
from keras import Model, Sequential
from keras import backend as K
from keras.activations import relu
from keras.callbacks import TensorBoard
from keras.initializers import random_normal
from keras.layers import Dense, Dropout, Input, TimeDistributed, Bidirectional, LSTM, Lambda, SimpleRNN, \
    BatchNormalization
from keras.optimizers import Adam
from keras.utils import get_custom_objects

from constants import TRAIN_TARGET_ROOT
from util.keras_util import ReportCallback, BatchGenerator
from util.train_util import get_num_features, get_corpus, get_target_dir

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
parser.add_argument('-e', '--num_epochs', type=int, nargs='?', default=20,
                    help=f'(optional) number of epochs to train the model (default: {20})')
parser.add_argument('--train_steps', type=int, nargs='?', default=0,
                    help=f'(optional) number of batches per epoch to use for training (default: all)')
parser.add_argument('--valid_steps', type=int, nargs='?', default=0,
                    help=f'(optional) number of batches per epoch to use for validation (default: all)')
parser.add_argument('--architecture', type=str, nargs='?', choices=['ds1', 'ds2', 'x'], default='ds1',
                    help=f'(optional) model architecture to use')
args = parser.parse_args()


def main():
    target_dir = get_target_dir('BRNN', args)
    print('loading train-/dev-/test-set')
    corpus = get_corpus(args)
    train_set, dev_set, test_set = corpus.train_dev_test_split()
    print(f'train/dev/test: {len(train_set)}/{len(dev_set)}/{len(test_set)} '
          f'({100*len(train_set)//len(corpus)}/{100*len(dev_set)//len(corpus)}/{100*len(test_set)//len(corpus)}%)')

    num_features = get_num_features(args.feature_type)
    model = create_model(args.architecture, num_features)
    model.summary()

    model, history = train_model(model, target_dir, train_set, dev_set)

    evaluate_model(model, test_set)

    model.save(os.path.join(target_dir, 'model.h5'))
    K.clear_session()


def create_model(architecture, num_features):
    get_custom_objects().update({"clipped_relu": clipped_relu})
    if architecture == 'ds1':
        return create_model_ds1(num_features)
    elif architecture == 'ds2':
        return create_model_ds2(num_features)
    return create_model_x(num_features)


def create_model_ds1(input_dim, fc_size=2048, rnn_size=512, output_dim=28):
    """
    DeepSpeech 1 model architecture with a few changes:
    - Feature type can be selected (number of features may therefore vary)
    - no language model integrated

    Reference: https://arxiv.org/abs/1412.5567

    :param input_dim: number of input features
    :param fc_size: number of units in FC layers
    :param rnn_size: number of units in recurrent layer
    :param output_dim: number of units in output layer (=number of target classes)
    :return:
    """

    init = random_normal(stddev=0.046875)

    model = Sequential()

    # input layer
    # input_data = Input(shape=(None, input_dim), name='the_input')
    model.add(BatchNormalization(axis=-1, input_shape=(None, input_dim), name='the'))

    # 3 FC layers with dropout
    model.add(TimeDistributed(
        Dense(fc_size, kernel_initializer=init, bias_initializer=init, activation=clipped_relu, name='FC_1')))
    model.add(Dropout(0.1))
    model.add(TimeDistributed(
        Dense(fc_size, kernel_initializer=init, bias_initializer=init, activation=clipped_relu, name='FC_2')))
    model.add(Dropout(0.1))
    model.add(TimeDistributed(
        Dense(fc_size, kernel_initializer=init, bias_initializer=init, activation=clipped_relu, name='FC_3')))
    model.add(Dropout(0.1))

    # RNN layer
    model.add(Bidirectional(
        SimpleRNN(rnn_size, return_sequences=True, activation=clipped_relu, kernel_initializer='he_normal',
                  name='BiRNN'), merge_mode='sum'))

    # Output layer
    model.add(TimeDistributed(Dense(output_dim, kernel_initializer=init, bias_initializer=init, activation='softmax'),
                              name='y_pred'))

    model_input = model.inputs[0]
    y_pred = model.outputs[0]

    labels = Input(name='the_labels', shape=[None, ], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, name='ctc')([y_pred, labels, input_length, label_length])
    # decoded_out = Lambda(ctc_decode_func, name='ctc_decoded')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[model_input, labels, input_length, label_length], outputs=loss_out)
    return model


def create_model_ds2(num_features, fc_size=2048, rnn_size=512, dropout=[0.1, 0.1, 0.1], output_dim=28):
    pass


def create_model_x(num_features, fc_size=2048, rnn_size=512, dropout=[0.1, 0.1, 0.1], output_dim=28):
    """
    Architecture from https://github.com/robmsmt/KerasDeepSpeech (ds1_dropout)
    - LSTM instead of Simple RNN
    - no BatchNorm
    - Dropout in RNN Layer
    """

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
    return model


def train_model(model, target_dir, train_set, dev_set):
    train_batches = BatchGenerator(train_set, args.feature_type, args.batch_size, args.train_steps)
    dev_batches = BatchGenerator(dev_set, args.feature_type, args.batch_size, args.valid_steps)

    print(f'Training on {len(train_batches)} batches ({len(train_batches.speech_segments)} speech segments)')
    print(f'Validating on {len(dev_batches)} batches ({len(dev_batches.speech_segments)} speech segments)')

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    model.compile(optimizer=opt, loss=ctc)

    cb_list = []
    tb_cb = TensorBoard(log_dir=target_dir, histogram_freq=1, write_graph=True, write_images=True)
    cb_list.append(tb_cb)

    y_pred = model.get_layer('ctc').input[0]
    input_data = model.get_layer('the_input').input
    report = K.function([input_data, K.learning_phase()], [y_pred])
    report_cb = ReportCallback(report, dev_batches, model, target_dir)
    # cb_list.append(report_cb)
    # create/add more callbacks here

    history = model.fit_generator(generator=train_batches.next_batch(),
                                  steps_per_epoch=train_batches.steps,
                                  epochs=args.num_epochs,
                                  callbacks=cb_list,
                                  validation_data=dev_batches.next_batch(),
                                  # validation_steps=dev_batches.steps,
                                  verbose=1
                                  )
    return model, history


def evaluate_model(model, test_set):
    test_batches = BatchGenerator(test_set, args.feature_type, args.batch_size)
    print(f'Evaluating on {len(test_batches)} batches ({len(test_batches.speech_segments)} speech segments)')
    # model.evaluate_generator(test_batches.next_batch(), steps=test_batches.steps)

    # for inputs, outputs in test_batches.next_batch():
    #     X_lengths = inputs['input_length']
    #     Y_pred = model.predict(inputs)
    #     res = tf.keras.backend.ctc_decode(Y_pred, X_lengths)
    #     print(f'prediction: {decode(y_pred)}')
    #     print(f'actual: {decode(y)}')


def clipped_relu(x):
    return relu(x, max_value=20)


def ctc(y_true, y_pred):
    return y_pred


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_decode_func(args):
    y_pred, labels, input_length, label_length = args
    sequences, probs = K.ctc_decode(y_pred, tf.reshape(input_length, [-1]), greedy=True)
    return tf.cast(sequences[0], tf.float32)


if __name__ == '__main__':
    main()
