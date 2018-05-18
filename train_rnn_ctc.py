# RNN implementation inspired by https://github.com/philipperemy/tensorflow-ctc-speech-recognition
import argparse
import os
import random
import time

import tensorflow as tf

# Some configs
from corpus_util import load_corpus
from file_logger import FileLogger
from log_util import log_prediction
from rnn_utils import create_x_y, CHAR_TOKENS, decode

parser = argparse.ArgumentParser(description="""Train RNN with CTC cost function for speech recognition""")
parser.add_argument('corpus', type=str, choices=['rl', 'ls'],
                    help='corpus on which to train the RNN (rl=ReadyLingua, ls=LibriSpeech')
parser.add_argument('language', type=str,
                    help='language on which to train the RNN')
args = parser.parse_args()

LS_SOURCE_ROOT = r'E:\librispeech-corpus' if os.name == 'nt' else '/media/all/D1/librispeech-corpus'
RL_SOURCE_ROOT = r'E:\readylingua-corpus' if os.name == 'nt' else '/media/all/D1/readylingua-corpus'
LS_TARGET_ROOT = r'E:\librispeech-data' if os.name == 'nt' else '/media/all/D1/librispeech-data'
RL_TARGET_ROOT = r'E:\readylingua-data' if os.name == 'nt' else '/media/all/D1/readylingua-data'

ls_corpus_file = os.path.join(LS_SOURCE_ROOT, 'librispeech.corpus')
rl_corpus_file = os.path.join(RL_SOURCE_ROOT, 'readylingua.corpus')

# Hyper-parameters
num_features = 13
# 26 lowercase ASCII chars + space + blank = 28 labels
num_classes = len(CHAR_TOKENS) + 2

# Hyper-parameters
num_epochs = 10000
num_hidden = 100
num_layers = 1
batch_size = 1

num_examples = 1
num_batches_per_epoch = int(num_examples / batch_size)

file_logger = FileLogger('out.tsv', ['curr_epoch', 'train_cost', 'train_ler', 'val_cost', 'val_ler'])


def train_rnn_ctc(corpus):
    corpus.summary()

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

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()

        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):
                x_train, y_train, ground_truth = next_training_batch()

                feed = {inputs: x_train, targets: y_train, seq_len: [x_train.shape[1]]}
                batch_cost, _ = session.run([cost, optimizer], feed)

                train_cost += batch_cost * batch_size
                train_ler += session.run(ler, feed_dict=feed) * batch_size

                # Decoding
                d = session.run(decoded[0], feed_dict=feed)
                str_decoded = decode(d[1])

                log_prediction(ground_truth, str_decoded, 'train-set')

            train_cost /= num_examples
            train_ler /= num_examples

            x_val, y_val, ground_truth = next_validation_batch()

            val_feed = {inputs: x_val, targets: y_val, seq_len: [x_val.shape[1]]}
            val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

            # Decoding
            d = session.run(decoded[0], feed_dict=val_feed)
            str_decoded = decode(d[1])

            log_prediction(ground_truth, str_decoded, 'dev-set')

            file_logger.write([curr_epoch + 1, train_cost, train_ler, val_cost, val_ler])

            log = f'=== Epoch {curr_epoch+1}/{num_epochs}, train_cost = {train_cost:.3f}, train_ler = {train_ler:.3f}, ' \
                  f'val_cost = {val_cost:.3f}, val_ler = {val_ler:.3f}, time = {time.time() - start:.3f} ==='
            print(log)


def next_training_batch():
    return next_batch(rl_train)


def next_validation_batch():
    return next_batch(rl_dev)


def next_batch(corpus_subset):
    random_entry = random.choice(corpus_subset)
    random_speech = random.choice(random_entry.speech_segments)
    rate, audio = random_speech.audio
    text = random_speech.text

    train_inputs, train_targets = create_x_y(audio, rate, text)
    return train_inputs, train_targets, text


if __name__ == "__main__":
    if args.corpus == 'rl':
        corpus = load_corpus(rl_corpus_file)
    elif args.corpus == 'ls':
        corpus = load_corpus(ls_corpus_file)
    corpus = corpus(languages=args.language, include_numeric=False)
    rl_train, rl_dev, rl_test = corpus.train_dev_test_split()
    train_rnn_ctc(corpus)
