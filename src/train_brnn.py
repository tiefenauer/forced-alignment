"""
Train a bi-directional Recurrent Neural Network (BRNN) similar to the model used for DeepSpeech (https://arxiv.org/abs/1412.5567)
The implementation was inspired by https://github.com/igormq/asr-study, with adjustments to work with Keras 2.x and
Python 3.

The type of features can be chosen (MFCC, Mel-Spectrogram or Power-Spectrogram). The script will look for a file named
`features_xxx.h5` (whereas xxx is one of 'mfcc', 'mel' or 'pow') which must contain precomputed features. If no such
file is found, the features will be calculated on the fly which significantly slows down training.

The following results will be written to a target folder:
- `model.h5`: Trained model including weights
- `events.out.tfevents...`: TensorBoard events file
- `history.pkl`: file containing the history of the training process
- `train.log`: complete log of the training process (containing everything that was printed to the console)

The training process can be adjusted by setting parameters. Type `python train_brnn.py -h` for help.

Sample call (training on the LibriSpeech corpus with MFCC as features):
    python train_brnn.py -c ls -f mfcc
"""
import argparse
import pickle
from os.path import join

import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from constants import TRAIN_ROOT, NUM_EPOCHS, BATCH_SIZE, FEATURE_TYPE, CORPUS, LANGUAGE, NUM_STEPS_TRAIN, \
    NUM_STEPS_VAL, ARCHITECTURE
from core.callbacks import ReportCallback
from util.brnn_util import deep_speech_model, generate_train_dev_test, ctc_dummy_loss, decoder_dummy_loss, ler
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
    description="""Train a bi-directional RNN with CTC cost function for speech recognition""")
parser.add_argument('-c', '--corpus', type=str, choices=['rl', 'ls'], nargs='?', default=CORPUS,
                    help=f'(optional) corpus on which to train (rl=ReadyLingua, ls=LibriSpeech). Default: {CORPUS}')
parser.add_argument('-l', '--language', type=str, nargs='?', default=LANGUAGE,
                    help=f'(optional) language on which to train the RNN. Default: {LANGUAGE}')
parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=BATCH_SIZE,
                    help=f'(optional) number of speech segments to include in one batch (default: {BATCH_SIZE})')
parser.add_argument('-f', '--feature_type', type=str, nargs='?', choices=['mfcc', 'mel', 'pow'], default=FEATURE_TYPE,
                    help=f'(optional) features to use for training (default: {FEATURE_TYPE})')
parser.add_argument('-t', '--target_root', type=str, nargs='?', default=TRAIN_ROOT,
                    help=f'(optional) root of folder where results will be written to (default: {TRAIN_ROOT})')
parser.add_argument('-e', '--num_epochs', type=int, nargs='?', default=NUM_EPOCHS,
                    help=f'(optional) number of epochs to train the model (default: {NUM_EPOCHS})')
parser.add_argument('--train_steps', type=int, nargs='?', default=NUM_STEPS_TRAIN,
                    help=f"""(optional) number of batches per epoch to use for training. A value of zero means all. 
                    Default: {NUM_STEPS_TRAIN}""")
parser.add_argument('--valid_steps', type=int, nargs='?', default=NUM_STEPS_VAL,
                    help=f"""(optional) number of batches per epoch to use for validation. A value of zero means all. 
                    Default: {NUM_STEPS_VAL}""")
parser.add_argument('--architecture', type=str, nargs='?', choices=['ds1', 'ds2', 'x'], default=ARCHITECTURE,
                    help=f"""(optional) model architecture to use (currently not used since only DeepSpeech is supported).
                    Default: {ARCHITECTURE}""")
args = parser.parse_args()


def main():
    target_dir = get_target_dir('BRNN', args)
    log_file_path = join(target_dir, 'train.log')
    redirect_to_file(log_file_path)
    print(f'Results will be written to: {target_dir}')

    corpus = get_corpus(args.corpus, args.language)
    print(f'training on corpus {corpus.name}')

    num_features = get_num_features(args.feature_type)
    print(f'number of features is: {num_features}')

    model = create_model(args.architecture, num_features)
    model.summary()

    train_it, val_it, test_it = generate_train_dev_test(corpus, args.language, args.feature_type, args.batch_size)
    total_n = train_it.n + val_it.n + test_it.n
    print(f'train/dev/test: {train_it.n}/{val_it.n}/{test_it.n} '
          f'({100*train_it.n//total_n}/{100*val_it.n//total_n}/{100*test_it.n//total_n}%)')
    history = train_model(model, target_dir, train_it, val_it)

    model.save(join(target_dir, 'model.h5'))
    with open(join(target_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    K.clear_session()


def create_model(architecture, num_features):
    """
    create uncompiled model with given architecture
    NOTE: currently only the DeepSpeech model is supported. Other models can be added here
    :param architecture: name of the architecture (see descriptions in argparse)
    :param num_features: number of features (hidden units in the input layer)
    :return:
    """
    return deep_speech_model(num_features)
    # if architecture == 'ds1':
    #     return deep_speech_model(num_features)
    # elif architecture == 'ds2':
    #     return create_model_ds2(num_features)
    # elif architecture == 'poc':
    #     return create_model_poc(num_features)
    # return create_model_x(num_features)


def train_model(model, target_dir, train_it, val_it):
    print(f'Batch size is {args.batch_size}')
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

    report_cb = ReportCallback(model, val_it, target_dir)
    cb_list.append(report_cb)

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


if __name__ == '__main__':
    main()
