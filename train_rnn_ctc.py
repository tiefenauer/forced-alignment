# RNN implementation inspired by https://github.com/philipperemy/tensorflow-ctc-speech-recognition
import argparse
import random
import time

import numpy as np
import tensorflow as tf
from os.path import exists

from corpus_util import load_corpus
from file_logger import FileLogger
from log_util import *
from rnn_utils import create_x_y, CHAR_TOKENS, decode, DummyCorpus

# -------------------------------------------------------------
# Constants, defaults and env-vars
# -------------------------------------------------------------
TARGET_ROOT = r'E:\\' if os.name == 'nt' else '/media/all/D1'  # default target directory
NUM_EPOCHS = 10000  # number of epochs to train on
NOW = datetime.now()
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# -------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="""Train RNN with CTC cost function for speech recognition""")
parser.add_argument('corpus', type=str, choices=['rl', 'ls'],
                    help='corpus on which to train the RNN (rl=ReadyLingua, ls=LibriSpeech')
parser.add_argument('language', type=str,
                    help='language on which to train the RNN')
parser.add_argument('-t', '--target_root', type=str, nargs='?', default=TARGET_ROOT,
                    help=f'(optional) root directory where results will be written to (default: {TARGET_ROOT})')
parser.add_argument('-e', '--num_epochs', type=int, nargs='?', default=NUM_EPOCHS,
                    help=f'(optional) number of epochs to train the model (default: {NUM_EPOCHS})')
parser.add_argument('-le', '--limit_entries', type=int, nargs='?',
                    help='(optional) number of corpus entries from training set to use for training (default: all)')
parser.add_argument('-ls', '--limit_segments', type=int, nargs='?',
                    help='(optional) number of aligned speech segments to use per corpus entry (default: all)')
args = parser.parse_args()

# -------------------------------------------------------------
# Other values
# -------------------------------------------------------------
ls_corpus_file = os.path.join(args.target_root, 'librispeech-corpus', 'librispeech.corpus')
rl_corpus_file = os.path.join(args.target_root, 'readylingua-corpus', 'readylingua.corpus')
target_dir = os.path.join(TARGET_ROOT, NOW.strftime('%Y-%m-%d-%H-%M-%S'))
print_to_file_and_console(target_dir)  # comment out to only log to console
print(f'Results will be written to: {os.path.abspath(target_dir)}')

# Hyper-parameters
num_features = 13
# 26 lowercase ASCII chars + space + blank = 28 labels
num_classes = len(CHAR_TOKENS) + 2

# Hyper-parameters
num_hidden = 100
num_layers = 1
max_shift = 2000  # maximum number of frames to shift the audio

# other options
batch_size = 100  # number of entries to process between validation


def main():
    print(create_args_str(args))

    if args.corpus == 'rl':
        corpus = load_corpus(rl_corpus_file)
    elif args.corpus == 'ls':
        corpus = load_corpus(ls_corpus_file)
    corpus = corpus(languages=args.language)
    corpus.summary()

    print('creating model')
    model_parms = create_model()

    train_set, dev_set, test_set = corpus.train_dev_test_split()

    if args.limit_entries:
        # for test purposes only: train only on first corpus entry
        repeat_samples = train_set[:args.limit_entries]
        train_set = DummyCorpus(repeat_samples, 1, num_segments=args.limit_segments)
        dev_set = DummyCorpus(repeat_samples, 1, num_segments=args.limit_segments)

    print(f'training on {len(train_set)} corpus entries with {args.limit_segments or "all"} segments each')
    save_path = train_model(model_parms, train_set, dev_set, test_set, target_dir)
    print(f'Model saved in path: {save_path}')


def create_model():
    graph = tf.Graph()
    with graph.as_default():
        # Input sequences
        # Has size [batch_size, max_step_size, num_features], but the
        # batch_size and max_step_size can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features], name='input')

        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # Sequence length: 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        # single LSTM cell
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

        shape = tf.shape(inputs)
        batch_s, max_time_steps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
        # Zero initialization
        # Tip: Is tf.zeros_initializer the same?
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)

        # optimizer = tf.train.AdamOptimizer().minimize(cost)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(cost)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    return {
        'graph': graph,
        'cost': cost,
        'optimizer': optimizer,
        'ler': ler,
        'inputs': inputs,
        'targets': targets,
        'seq_len': seq_len,
        'decoded': decoded
    }


def train_model(model_parms, train_set, dev_set, test_set, target_dir):
    graph = model_parms['graph']
    cost = model_parms['cost']
    optimizer = model_parms['optimizer']
    ler = model_parms['ler']
    inputs = model_parms['inputs']
    targets = model_parms['targets']
    seq_len = model_parms['seq_len']
    decoded = model_parms['decoded']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    cost_logger = create_cost_logger(target_dir)
    epoch_logger = create_epoch_logger(target_dir)

    with tf.Session(graph=graph, config=config) as session:
        tf.global_variables_initializer().run()

        for curr_epoch in range(args.num_epochs):

            train_cost = train_ler = 0
            start = time.time()

            for x_train, y_train, ground_truth in generate_data(train_set, True):
                feed = {inputs: x_train, targets: y_train, seq_len: [x_train.shape[1]]}
                batch_cost, _ = session.run([cost, optimizer], feed)

                train_cost += batch_cost
                train_ler += session.run(ler, feed_dict=feed)

                # Decoding
                d = session.run(decoded[0], feed_dict=feed)
                prediction = decode(d[1])

                print_prediction(ground_truth, prediction, 'train-set')

            validation_data = generate_data(dev_set, True)
            x_val, y_val, ground_truth = random.choice(list(validation_data))

            val_feed = {inputs: x_val, targets: y_val, seq_len: [x_val.shape[1]]}
            val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

            # Decoding
            d = session.run(decoded[0], feed_dict=val_feed)
            prediction = decode(d[1])

            cost_logger.write_tabbed([curr_epoch + 1, train_cost, train_ler, val_cost, val_ler])

            val_str = f'=== Epoch {curr_epoch}, train_cost = {train_cost:.3f}, train_ler = {train_ler:.3f}, ' \
                      f'val_cost = {val_cost:.3f}, val_ler = {val_ler:.3f}, time = {time.time() - start:.3f} ==='
            print(val_str)
            print_prediction(ground_truth, prediction, 'dev-set')
            if curr_epoch % 20 == 0:
                epoch_logger.write(val_str)
                log_prediction(epoch_logger, ground_truth, prediction, 'dev_set')

        saver = tf.train.Saver()
        save_path = saver.save(session, os.path.join(target_dir, 'model.ckpt'))

    return save_path


def generate_data(corpus_entries, shift_audio):
    for corpus_entry in corpus_entries:
        segments_with_text = [speech for speech in corpus_entry.speech_segments_not_numeric if speech.text]
        for speech_segment in segments_with_text:
            rate, audio = speech_segment.audio
            ground_truth = speech_segment.text

            if shift_audio:
                shift = np.random.randint(low=1, high=max_shift)
                audio = audio[shift:]

            x, y = create_x_y(audio, rate, ground_truth)
            yield x, y, ground_truth


def create_cost_logger(log_dir):
    cost_logger = create_file_logger(log_dir, 'stats.tsv')
    cost_logger.write_tabbed(['epoch', 'train_cost', 'train_ler', 'val_cost', 'val_ler'])
    return cost_logger


def create_epoch_logger(log_dir):
    epoch_logger = create_file_logger(log_dir, 'epochs.txt')
    revision, branch_name, timestamp, message = get_commit()
    epoch_logger.write(f'----------------------------------')
    epoch_logger.write(f'Date: {NOW}')
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
