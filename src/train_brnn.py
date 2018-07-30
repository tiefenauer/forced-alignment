# https://github.com/robmsmt/KerasDeepSpeech
import argparse
import os
import pickle
from os import listdir
from os.path import splitext, join

import h5py
import tensorflow as tf
from keras import backend as K
from keras.activations import relu
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Input, TimeDistributed, Bidirectional, SimpleRNN, Activation
from keras.optimizers import Adam
from keras.utils import get_custom_objects

from constants import TRAIN_ROOT
from core.ctc_util import ctc_model
from core.dataset_generator import OnTheFlyFeaturesIterator, HFS5BatchGenerator
from util.corpus_util import get_corpus
from util.log_util import redirect_to_file
from util.train_util import get_num_features, get_target_dir

# -------------------------------------------------------------
# some Keras/TF setup
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "2"
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
# -------------------------------------------------------------

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
    log_file_path = os.path.join(target_dir, 'train.log')
    redirect_to_file(log_file_path)  # comment out to only log to console
    print(f'Results will be written to: {target_dir}')

    corpus = get_corpus(args.corpus, args.language)
    print(f'training on corpus {corpus.name}')

    num_features = get_num_features(args.feature_type)
    print(f'number of features is: {num_features}')

    model = create_model(args.architecture, num_features)
    model.summary()

    train_it, val_it, test_it = create_train_dev_test(corpus, args.language, args.feature_type, args.batch_size)
    total_n = train_it.n + val_it.n + test_it.n
    print(f'train/dev/test: {train_it.n}/{val_it.n}/{test_it.n} '
          f'({100*train_it.n//total_n}/{100*val_it.n//total_n}/{100*test_it.n//total_n}%)')
    history = train_model(model, target_dir, train_it, val_it)

    model.save(os.path.join(target_dir, 'model.h5'))
    with open(os.path.join(target_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    K.clear_session()


def create_train_dev_test(corpus, language, feature_type, batch_size):
    h5_features = list(join(corpus.root_path, file) for file in listdir(corpus.root_path)
                       if splitext(file)[0].startswith('features')
                       and feature_type in splitext(file)[0]
                       and splitext(file)[1] == '.h5')
    default_featuers = f'features_{feature_type}.h5'
    feature_file = default_featuers if default_featuers in h5_features else h5_features[0] if h5_features else None
    if feature_file:
        print(f'found precomputed features: {feature_file}. Using HDF5-Features')
        f = h5py.File(feature_file, 'r')
        # hack for RL corpus: because there are no train/dev/test-subsets train/validate/test on same subset
        train_ds = f['train'][language] if args.corpus == 'ls' else f['generic'][language]
        dev_ds = f['dev'][language] if args.corpus == 'ls' else f['generic'][language]
        test_ds = f['test'][language] if args.corpus == 'ls' else f['generic'][language]
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
    o = TimeDistributed(Dense(num_classes, name='y_pred', activation='softmax'), name='out')(o)

    return ctc_model(x, o)


def train_model(model, target_dir, train_it, val_it):
    print(f'Training on {len(train_it)} batches over {train_it.n} speech segments')
    print(f'Validating on {len(val_it)} batches over {val_it.n} speech segments')

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

    # hack: avoid logging the dummy losses
    # keras.callbacks.ProgbarLogger = CustomProgbarLogger

    history = model.fit_generator(generator=train_it,
                                  validation_data=val_it,
                                  # steps_per_epoch=train_it.n,
                                  epochs=args.num_epochs,
                                  callbacks=cb_list,
                                  # validation_steps=dev_batches.num_batches,
                                  verbose=1
                                  )
    return history


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
    return tf.reduce_mean(tf.edit_distance(y_pred, y_true, normalize=True, **kwargs))


if __name__ == '__main__':
    main()
