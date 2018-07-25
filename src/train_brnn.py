# https://github.com/robmsmt/KerasDeepSpeech
import argparse
import os
import pickle

import keras
import tensorflow as tf
from keras import backend as K
from keras.activations import relu
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Input, TimeDistributed, Bidirectional, SimpleRNN, Activation
from keras.optimizers import Adam
from keras.utils import get_custom_objects

from constants import TRAIN_ROOT
from core.callbacks import CustomProgbarLogger, ReportCallback
from core.ctc_util import ctc_model
from core.dataset_generator import BatchGenerator
from util.corpus_util import get_corpus
from util.train_util import get_num_features, get_target_dir

# some Keras/TF setup
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "2"
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

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
parser.add_argument('-t', '--target_root', type=str, nargs='?', default=TRAIN_ROOT,
                    help=f'(optional) root directory where results will be written to (default: {TRAIN_ROOT})')
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
    corpus = get_corpus(args.corpus)
    train_set, dev_set, test_set = corpus.train_dev_test_split()
    print(f'train/dev/test: {len(train_set)}/{len(dev_set)}/{len(test_set)} '
          f'({100*len(train_set)//len(corpus)}/{100*len(dev_set)//len(corpus)}/{100*len(test_set)//len(corpus)}%)')

    num_features = get_num_features(args.feature_type)
    model = create_model(args.architecture, num_features)
    model.summary()

    history = train_model(model, target_dir, train_set, dev_set)

    evaluate_model(model, test_set)

    model.save(os.path.join(target_dir, 'model.h5'))
    with open(os.path.join(target_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    K.clear_session()


def create_model(architecture, num_features):
    """
    create uncompiled model with given architecture
    NOTE: currently only the DeepSpeech model is supported. Other models can be added here
    :param architecture: name of the architecture (see descriptions in argparse)
    :param num_features: number of features in the input layer
    :return:
    """
    get_custom_objects().update({"clipped_relu": clipped_relu})
    return deep_speech_model(num_features)
    # if architecture == 'ds1':
    #     return deep_speech_model(num_features)
    # elif architecture == 'ds2':
    #     return create_model_ds2(num_features)
    # elif architecture == 'poc':
    #     return create_model_poc(num_features)
    # return create_model_x(num_features)


def deep_speech_model(num_features, num_hidden=2048, dropout=0.1, num_classes=28):
    """
    Deep Speech model with architecture as described in the paper:
        5 Layers (3xFC + 1xBRNN + 1xFC) with Dropout applied to FC layer
    The output contains 28 classes: a..z, space, blank

    Differences to the original setup:
        * We are not translating the raw audio files by 5 ms (Sec 2.1 in [1])
        * We are not striding the RNN to halve the timesteps (Sec 3.3 in [1])
        * We are not using frames of context

    Reference: [1] https://arxiv.org/abs/1412.5567
    """
    x = Input(name='inputs', shape=(None, num_features))
    o = x

    # First layer
    o = TimeDistributed(Dense(num_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Second layer
    o = TimeDistributed(Dense(num_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Third layer
    o = TimeDistributed(Dense(num_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Fourth layer
    o = Bidirectional(SimpleRNN(num_hidden, return_sequences=True,
                                dropout=dropout,
                                activation=clipped_relu,
                                kernel_initializer='he_normal'), merge_mode='sum')(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Fifth layer
    o = TimeDistributed(Dense(num_hidden))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = TimeDistributed(Dropout(dropout))(o)

    # Output layer
    o = TimeDistributed(Dense(num_classes))(o)

    return ctc_model(x, o)


def train_model(model, target_dir, train_set, dev_set):
    train_batches = BatchGenerator(train_set, args.feature_type, args.batch_size, args.train_steps)
    dev_batches = BatchGenerator(dev_set, args.feature_type, args.batch_size, args.valid_steps)

    print(f'Training on {len(train_batches)} batches over {len(train_batches.speech_segments)} speech segments')
    print(f'Validating on {len(dev_batches)} batches over {len(dev_batches.speech_segments)} speech segments')

    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    model.compile(loss={'ctc': ctc_dummy_loss, 'decoder': decoder_dummy_loss},
                  optimizer=opt,
                  metrics={'decoder': ler},  #
                  loss_weights=[1, 0]  # only optimize CTC cost
                  )

    # callbacks
    cb_list = []
    tb_cb = TensorBoard(log_dir=target_dir, write_graph=True, write_images=True)
    cb_list.append(tb_cb)

    # y_pred = model.get_layer('ctc').input[0]
    # input_data = model.get_layer('the_input').input
    # report = K.function([input_data, K.learning_phase()], [y_pred])
    # report_cb = ReportCallback(report, dev_batches, model, target_dir)
    # cb_list.append(report_cb)

    # model_ckpt = MetaCheckpoint(join(target_dir, 'model.h5'), training_args=args, meta=meta)
    # cb_list.append(model_ckpt)
    # best_ckpt = MetaCheckpoint(join(target_dir, 'best.h5'), monitor='val_decoder_ler',
    #                            save_best_only=True, mode='min', training_args=args, meta=meta)
    # cb_list.append(best_ckpt)
    # create/add more callbacks here

    # avoid logging the dummy losses
    # keras.callbacks.ProgbarLogger = lambda count_mode, stateful_metrics: CustomProgbarLogger(
    #     stateful_metrics=['loss', 'decoder_ler', 'val_loss', 'val_decoder_ler'])

    history = model.fit_generator(generator=train_batches.generate_batches(),
                                  steps_per_epoch=train_batches.num_batches,
                                  epochs=args.num_epochs,
                                  callbacks=cb_list,
                                  validation_data=dev_batches.generate_batches(),
                                  validation_steps=dev_batches.num_batches,
                                  verbose=1
                                  )
    return history


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


def ctc_dummy_loss(y_true, y_pred):
    """
    CTC-Loss for Keras. Because Keras has no CTC-loss built-in, we create this dummy.
    We simply use the output of the RNN, which corresponds to the CTC loss.
    """
    return y_pred


def decoder_dummy_loss(y_true, y_pred):
    """
    Loss of the decoded sequence. Since this is not optimized, we simply return a zero-Tensor
    """
    return K.zeros((1,))


def ler(y_true, y_pred, **kwargs):
    """
    LER-Loss (see https://www.tensorflow.org/api_docs/python/tf/edit_distance)
    """
    return tf.reduce_mean(tf.edit_distance(y_pred, y_true, **kwargs))


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_decode_func(args):
    y_pred, labels, input_length, label_length = args
    decoded, log_prob = tf.nn.ctc_greedy_decoder(y_pred, tf.reshape(input_length, [-1]))
    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(y_pred, tf.reshape(input_length, [-1]))
    # sparse = dense_to_sparse(decoded[0])
    # ed = tf.edit_distance(tf.cast(sparse, tf.int32), labels)
    # ler = tf.reduce_mean(ed)
    # return tf.sparse_to_dense(ler)
    # sequences, probs = K.ctc_decode(y_pred, tf.reshape(input_length, [-1]), greedy=False)
    # return tf.cast(sequences[0], tf.float32)


if __name__ == '__main__':
    main()
