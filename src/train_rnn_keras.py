# RNN implementation inspired by https://github.com/philipperemy/tensorflow-ctc-speech-recognition
import argparse
import random
import time

import collections
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras import backend as K
from keras.layers import LSTM, TimeDistributed
from keras.optimizers import TFOptimizer
from os.path import exists

from constants import TRAIN_TARGET_ROOT
from util.audio_util import distort, shift
from util.log_util import *
from util.plot_util import visualize_cost
from util.rnn_util import CHAR_TOKENS, decode, DummyCorpus, FileLogger, encode, pad_sequences, sparse_tuple_from
from util.train_util import get_poc, get_target_dir, get_num_features, get_corpus

# -------------------------------------------------------------
# Constants, defaults and env-vars
# -------------------------------------------------------------
BATCH_SIZE = 50
MAX_EPOCHS = 10000  # number of epochs to train on
LER_CONVERGENCE = 0.05  # LER value for convergence (average over last 10 epochs)
MAX_SHIFT = 2000  # maximum number of frames to shift the audio
FEATURE_TYPE = 'mfcc'
SYNTHESIZE = False
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# -------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="""Train RNN with CTC cost function for speech recognition""")
parser.add_argument('-p', '--poc', type=str, nargs='?',
                    help='(optional) PoC # to train. If used, a preset choice of parameters is used.')
parser.add_argument('-c', '--corpus', type=str, choices=['rl', 'ls'],
                    help='corpus on which to train the RNN (rl=ReadyLingua, ls=LibriSpeech')
parser.add_argument('-l', '--language', type=str,
                    help='language on which to train the RNN')
parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=BATCH_SIZE,
                    help=f'(optional) number of speech segments to include in one batch (default:{BATCH_SIZE})')
parser.add_argument('-f', '--feature_type', type=str, nargs='?', choices=['mfcc', 'mel', 'pow'], default='mfcc',
                    help=f'(optional) features to use for training (default: {FEATURE_TYPE})')
parser.add_argument('-id', '--id', type=str, nargs='?',
                    help='(optional) specify ID of single corpus entry on which to train on')
parser.add_argument('-ix', '--ix', type=str, nargs='?',
                    help='(optional) specify index of single corpus entry on which to train on')
parser.add_argument('-s', '--synthesize', action='store_true', default=SYNTHESIZE,
                    help=f'(optional) synthesize audio for training by adding distortion (default: {SYNTHESIZE})')
parser.add_argument('-t', '--target_root', type=str, nargs='?', default=TRAIN_TARGET_ROOT,
                    help=f'(optional) root directory where results will be written to (default: {TRAIN_TARGET_ROOT})')
parser.add_argument('-e', '--num_epochs', type=int, nargs='?', default=MAX_EPOCHS,
                    help=f'(optional) number of epochs to train the model (default: {MAX_EPOCHS})')
parser.add_argument('-le', '--limit_entries', type=int, nargs='?',
                    help='(optional) number of corpus entries from training set to use for training (default: all)')
parser.add_argument('-ls', '--limit_segments', type=int, nargs='?',
                    help='(optional) number of aligned speech segments to use per corpus entry (default: all)')
args = get_poc(parser.parse_args())

# -------------------------------------------------------------
# Other values
# -------------------------------------------------------------
num_classes = len(CHAR_TOKENS) + 2  # 26 lowercase ASCII chars + space + blank = 28 labels
num_hidden = 100
num_layers = 1


def main():
    target_dir = get_target_dir('RNN_KERAS', args)
    num_features = get_num_features(args.feature_type)
    corpus = get_corpus(args)

    train_set, dev_set, test_set = create_train_dev_test(args, corpus)

    model_parms = create_model(num_features)
    train_model(model_parms, target_dir, train_set, dev_set, test_set)

    fig_ctc, fig_ler, _ = visualize_cost(target_dir, args)
    fig_ctc.savefig(os.path.join(target_dir, 'ctc_cost.png'), bbox_inches='tight')
    fig_ler.savefig(os.path.join(target_dir, 'ler_cost.png'), bbox_inches='tight')


def create_train_dev_test(args, corpus):
    repeat_sample = None

    if args.id is not None:
        if args.id not in corpus.keys:
            print(f'Error: no entry with id={args.id} found!')
            return exit()

        print(f'training on corpus entry with id={args.id}')
        repeat_sample = corpus[args.id]

    if args.ix is not None:
        if args.ix > len(corpus):
            print(f'Error: {args.id} exceeds corpus bounds ({len(corpus)} entries)!')
            return exit()
        print(f'training on corpus entry with index={args.ix}')
        repeat_sample = corpus[args.ix]

    if repeat_sample:
        train_set = DummyCorpus([repeat_sample], 1, args.limit_segments)
        dev_set = DummyCorpus([repeat_sample], 1, args.limit_segments)
        test_set = DummyCorpus([repeat_sample], 1, args.limit_segments)
        return train_set, dev_set, test_set

    train_set, dev_set, test_set = corpus.train_dev_test_split()
    if args.limit_entries:
        print(f'limiting corpus entries to {args.limit_entries}')
        repeat_samples = train_set[:args.limit_entries]
        train_set = DummyCorpus(repeat_samples, 1, num_segments=args.limit_segments)
        dev_set = DummyCorpus(repeat_samples, 1, num_segments=args.limit_segments)

    return train_set, dev_set, test_set


def create_model(num_features):
    print('creating model')
    # inputs = tf.placeholder(tf.float32, [None, None, num_features], name='input')
    inputs = Input(shape=(None, num_features), name='the_input')

    # targets = tf.sparse_placeholder(tf.int32, name='targets')
    targets = tf.sparse_placeholder(tf.int32, name='targets')

    # seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
    seq_len = tf.placeholder(tf.int32, [None], name='input_length')

    # cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    # stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    # outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    outputs = TimeDistributed(LSTM(num_hidden, return_sequences=True))(inputs)

    # shape = tf.shape(inputs)
    # batch_s, max_time_steps = shape[0], shape[1]
    shape = K.shape(inputs)
    batch_s, max_time_steps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])

    # W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    W = K.variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    # b = tf.Variable(tf.constant(0., shape=[num_classes]))
    b = K.variable(K.constant(0, shape=[num_classes]))

    # logits = tf.matmul(outputs, W) + b
    logits = K.dot(outputs, W) + b

    # logits = tf.reshape(logits, [batch_s, -1, num_classes])
    logits = K.reshape(logits, [batch_s, -1, num_classes])
    # logits = tf.transpose(logits, (1, 0, 2))
    logits = tf.transpose(logits, (1, 0, 2))

    # loss = tf.nn.ctc_loss(targets, logits, seq_len)
    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    # cost = tf.reduce_mean(loss)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9).minimize(cost)
    # will be fed to compile

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    return {
        'num_features': num_features,
        'cost': cost,
        'ler': ler,
        'inputs': inputs,
        'targets': targets,
        'seq_len': seq_len,
        'decoded': decoded,
        'log_prob': log_prob
    }


def train_model(model_parms, target_dir, train_set, dev_set, test_set):
    print(f'training on {len(train_set)} corpus entries with {args.limit_segments or "all"} segments each')
    num_features = model_parms['num_features']
    cost = model_parms['cost']
    ler = model_parms['ler']
    inputs = model_parms['inputs']
    targets = model_parms['targets']
    seq_len = model_parms['seq_len']
    decoded = model_parms['decoded']

    train_cost_logger = create_cost_logger(target_dir, 'stats.tsv')
    epoch_logger = create_epoch_logger(target_dir)

    model = Model(inputs=[inputs, targets, seq_len], outputs=[decoded, ler, cost])
    opt = TFOptimizer(tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9))
    model.compile(optimizer=opt, loss=cost)

    with tf.Session(graph=graph, config=config) as session:
        tf.global_variables_initializer().run()

        curr_epoch = 0

        # sliding window over the values for CTC- and LER-cost of the last 10 epochs (train-set)
        train_ctcs = collections.deque(maxlen=10)
        train_lers = collections.deque(maxlen=10)

        # sliding vindow over the last 10 average values of CTC/LER-cost (train-set)
        ler_train_means = collections.deque([0], maxlen=10)

        # sliding window over the LER values of the last 10 epochs (dev-set)
        val_ctcs = collections.deque(maxlen=10)
        val_lers = collections.deque(maxlen=10)

        convergence = False
        # train until convergence or MAX_EPOCHS
        while not convergence and curr_epoch < MAX_EPOCHS:
            add_distortion = curr_epoch > 0 and args.synthesize
            curr_epoch += 1
            num_samples = 0
            ctc_train = ler_train = 0
            start = time.time()

            # iterate over batches for current epoch
            for X, Y, batch_seq_len, ground_truths in generate_batches(train_set, num_features,
                                                                       distort_audio=add_distortion):
                feed = {inputs: X, targets: Y, seq_len: batch_seq_len}
                batch_cost, _ = session.run([cost, opt], feed)

                batch_len = X.shape[0]
                ctc_train += batch_cost * batch_len
                ler_train += session.run(ler, feed_dict=feed) * batch_len

                # Decoding
                d = session.run(decoded[0], feed_dict=feed)
                dense_decoded = tf.sparse_tensor_to_dense(d, default_value=0).eval(session=session)

                for i, prediction_enc in enumerate(dense_decoded):
                    ground_truth = ground_truths[i]
                    prediction = decode(prediction_enc)
                    print_prediction(ground_truth, prediction, 'train-set')
                    log_prediction(epoch_logger, ground_truth, prediction, 'train_set')

                num_samples += batch_len

            # calculate costs for current epoch
            ctc_train /= num_samples
            ler_train /= num_samples
            train_ctcs.append(ctc_train)
            train_lers.append(ler_train)

            # update means
            ctc_train_mean = np.array(train_ctcs).mean()
            ler_train_mean = np.array(train_lers).mean()
            ler_train_means.append(ler_train_mean)

            # convergence reached if mean LER-rate is below threshold and mean LER change rate is below 1%
            ler_diff = np.diff(ler_train_means).mean()
            convergence = ler_train_mean < LER_CONVERGENCE and abs(ler_diff) < 0.01

            # validate cost with a randomly chosen entry from the dev-set that has been randomly shifted
            X_val, Y_val, val_seq_len, val_ground_truths = random.choice(list(generate_batches(dev_set, True)))
            val_feed = {inputs: X_val, targets: Y_val, seq_len: val_seq_len}
            ctc_val, ler_val = session.run([cost, ler], feed_dict=val_feed)
            val_ctcs.append(ctc_val)
            ctc_val_mean = np.array(val_ctcs).mean()
            val_lers.append(ler_val)
            ler_val_mean = np.array(val_lers).mean()

            train_cost_logger.write_tabbed(
                [curr_epoch, ctc_train, ctc_train_mean, ler_train, ler_train_mean, ctc_val, ctc_val_mean, ler_val,
                 ler_val_mean])

            val_str = f'=== Epoch {curr_epoch}, ctc_train = {ctc_train:.3f}, ler_train = {ler_train:.3f}, ' \
                      f'ctc_val = {ctc_val:.3f}, ler_val = {ler_val:.3f}, time = {time.time() - start:.3f}, ' \
                      f'ctc_mean = {ctc_train_mean}, ler_mean = {ler_train_mean:.3f}, ler_diff = {ler_diff:.3f} ==='
            print(val_str)
            epoch_logger.write(val_str)

        print(f'convergence reached after {curr_epoch} epochs!')
        saver = tf.train.Saver()
        save_path = saver.save(session, os.path.join(target_dir, 'model.ckpt'))
        print(f'Model saved to path: {save_path}')

    return save_path


def generate_batches(corpus_entries, num_features, shift_audio=False, distort_audio=False):
    speech_segments = list(seg for corpus_entry in corpus_entries for seg in corpus_entry.speech_segments_not_numeric)
    l = len(speech_segments)
    for ndx in range(0, l, args.batch_size):
        batch = []
        for speech_segment in speech_segments[ndx:min(ndx + args.batch_size, l)]:
            audio = speech_segment.audio  # save original audio signal
            if distort_audio:
                speech_segment.audio = distort(audio, speech_segment.rate, tempo=True)
            if shift_audio:
                speech_segment.audio = shift(audio)  # clip audio before calculating MFCC/spectrogram

            if args.feature_type == 'mfcc':
                features = speech_segment.mfcc()
            elif args.feature_type == 'mel':
                features = speech_segment.mel_specgram(num_features).T
            else:
                features = speech_segment.power_specgram().T
            speech_segment.audio = audio  # restore original audio for next epoch

            batch.append((features, speech_segment.text))

        features, ground_truths = zip(*batch)

        X, batch_seq_len = pad_sequences(features)
        Y = sparse_tuple_from([encode(truth) for truth in ground_truths])

        yield X, Y, batch_seq_len, ground_truths


def create_cost_logger(log_dir, log_file):
    cost_logger = create_file_logger(log_dir, log_file)
    cost_logger.write_tabbed(
        ['epoch', 'ctc_train', 'ctc_train_mean', 'ler_train', 'ler_train_mean', 'ctc_val', 'ler_val', 'ler_val_mean'])
    return cost_logger


def create_epoch_logger(log_dir):
    epoch_logger = create_file_logger(log_dir, 'epochs.txt')
    revision, branch_name, timestamp, message = get_commit()
    epoch_logger.write(f'----------------------------------')
    epoch_logger.write(f'Directory: {log_dir}')
    epoch_logger.write(f'Branch: {branch_name}')
    epoch_logger.write(f'Commit: {revision} ({timestamp}, {message})')
    epoch_logger.write(f'----------------------------------')
    return epoch_logger


def create_file_logger(log_dir, file_name):
    if not exists(log_dir):
        makedirs(log_dir)
    file_path = os.path.join(log_dir, file_name)
    file_logger = FileLogger(file_path)
    return file_logger


if __name__ == "__main__":
    main()
