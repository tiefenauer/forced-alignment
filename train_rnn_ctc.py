# RNN implementation inspired by https://github.com/philipperemy/tensorflow-ctc-speech-recognition
import random
import time

import numpy as np
import tensorflow as tf

# Some configs
from corpus_util import load_corpus
from file_logger import FileLogger
from rnn_utils import FIRST_INDEX, convert_inputs_to_ctc_format
from log_util import log_prediction

num_features = 13
# Accounting the 0th index +  space + blank label + 3 umlauts = 31 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1 + 3

# Hyper-parameters
num_epochs = 10000
num_hidden = 100
num_layers = 1
batch_size = 1

num_examples = 1
num_batches_per_epoch = int(num_examples / batch_size)

file_logger = FileLogger('out.tsv', ['curr_epoch', 'train_cost', 'train_ler', 'val_cost', 'val_ler'])

rl_corpus = load_corpus(r'E:\readylingua-corpus\readylingua.corpus')
rl_train, rl_dev, rl_test = rl_corpus.train_dev_test_split()


def train_rnn_ctc():
    graph = tf.Graph()
    with graph.as_default():
        # Input sequences
        # Has size [batch_size, max_step_size, num_features], but the
        # batch_size and max_step_size can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features])

        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32)

        # Sequence length: 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None])

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
                train_inputs, train_targets, train_seq_len, train_txt, original_txt = next_training_batch()
                feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}

                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost * batch_size
                train_ler += session.run(ler, feed_dict=feed) * batch_size

                # Decoding
                d = session.run(decoded[0], feed_dict=feed)
                str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
                # Replacing blank label to none
                str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
                # Replacing space label to space
                str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

                log_prediction(original_txt, train_txt, str_decoded, 'train-set')

            train_cost /= num_examples
            train_ler /= num_examples

            val_inputs, val_targets, val_seq_len, val_txt, original_txt = next_validation_batch()
            val_feed = {inputs: val_inputs, targets: val_targets, seq_len: val_seq_len}

            val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

            # Decoding
            d = session.run(decoded[0], feed_dict=val_feed)
            str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
            # Replacing blank label to none
            str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
            # Replacing space label to space
            str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

            log_prediction(original_txt, val_txt, str_decoded, 'dev-set')

            file_logger.write([curr_epoch + 1, train_cost, train_ler, val_cost, val_ler])

            log = f'=== Epoch {curr_epoch+1}/{num_epochs}, train_cost = {train_cost:.3f}, train_ler = {train_ler:.3f}, ' \
                  f'val_cost = {val_cost:.3f}, val_ler = {val_ler:.3f}, time = {time.time() - start:.3f} ==='
            print(log)


def next_training_batch():
    return next_batch(rl_train)


def next_validation_batch():
    return next_batch(rl_test)


def next_batch(corpus_subset):
    random_entry = random.choice(corpus_subset)
    random_speech = random.choice(random_entry.speech_segments)
    rate, audio = random_speech.audio
    text = random_speech.text

    train_inputs, train_targets, train_seq_len, input_text = convert_inputs_to_ctc_format(audio, rate, text)
    return train_inputs, train_targets, train_seq_len, input_text, text


if __name__ == "__main__":
    train_rnn_ctc()
