# RNN implementation inspired by https://github.com/philipperemy/tensorflow-ctc-speech-recognition
import argparse
import math
import random
import time

import collections
import numpy as np
import tensorflow as tf
from os.path import exists

from definitions import TRAIN_TARGET_ROOT
from util.audio_util import distort, shift
from util.corpus_util import load_corpus
from util.log_util import *
from util.plot_util import visualize_cost
from util.rnn_util import CHAR_TOKENS, decode, DummyCorpus, FileLogger, encode, pad_sequences, sparse_tuple_from

# -------------------------------------------------------------
# Constants, defaults and env-vars
# -------------------------------------------------------------
BATCH_SIZE = 50
NUM_EPOCHS = 10000  # number of epochs to train on
LER_CONVERGENCE = 0.05  # LER value for convergence (average over last 10 epochs)
NOW = datetime.now()
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
parser.add_argument('-f', '--feature_type', type=str, nargs='?', choices=['mfcc', 'spec'], default='mfcc',
                    help=f'(optional) features to use for training (default: {FEATURE_TYPE})')
parser.add_argument('-id', '--id', type=str, nargs='?',
                    help='(optional) specify ID of single corpus entry on which to train on')
parser.add_argument('-ix', '--ix', type=str, nargs='?',
                    help='(optional) specify index of single corpus entry on which to train on')
parser.add_argument('-s', '--synthesize', action='store_true', default=SYNTHESIZE,
                    help=f'(optional) synthesize audio for training by adding distortion (default: {SYNTHESIZE})')
parser.add_argument('-t', '--target_root', type=str, nargs='?', default=TRAIN_TARGET_ROOT,
                    help=f'(optional) root directory where results will be written to (default: {TRAIN_TARGET_ROOT})')
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
ls_corpus_root = os.path.join(args.target_root, 'librispeech-corpus')
rl_corpus_root = os.path.join(args.target_root, 'readylingua-corpus')

# Hyper-parameters
num_features = 161 if args.feature_type == 'spec' else 13
num_classes = len(CHAR_TOKENS) + 2  # 26 lowercase ASCII chars + space + blank = 28 labels
num_hidden = 100
num_layers = 1

# other
batch_size = args.batch_size
# batch_size = 1

# PoC parameters
poc_args = {
    'poc_1': {
        'corpus': 'rl',
        'language': 'de',
        'id': 'andiefreudehokohnerauschenrein',
        'limit_segments': 5
    },
    'poc_2': {
        'corpus': 'rl',
        'language': 'de',
        'id': 'andiefreudehokohnerauschenrein',
        'feature_type': 'spec',
        'limit_segments': None  # all
    },
    'poc_3': {
        'corpus': 'ls',
        'language': 'en',
        'id': 'tbd',
        'limit_segments': 5
    },
    'poc_4': {
        'corpus': 'ls',
        'language': 'en',
        'id': 'tbd',
        'limit_segments': 5
    }
}

target_dir = os.path.join(args.target_root, NOW.strftime('%Y-%m-%d-%H-%M-%S'))


def set_poc():
    """overrides the CLI args by preset values if PoC argument is set."""
    global target_dir
    print(create_args_str(args))
    if not args.poc:
        return
    poc = poc_args['poc_' + args.poc]
    for key, value in poc.items():
        setattr(args, key, value)
    print(create_args_str(args))

    target_dir += '_'.join(
        ['_poc_' + args.poc, args.language, args.feature_type, 'synthesized' if args.synthesize else 'original'])
    print(f'Results will be written to: {target_dir}')

    log_file_path = os.path.join(target_dir, 'train.log')
    print_to_file_and_console(log_file_path)  # comment out to only log to console


def main():
    set_poc()

    if args.corpus == 'rl':
        corpus = load_corpus(rl_corpus_root)
    elif args.corpus == 'ls':
        corpus = load_corpus(ls_corpus_root)
    corpus = corpus(languages=args.language)
    corpus.summary()

    train_set, dev_set, test_set = create_train_dev_test(args, corpus)

    print('creating model')
    model_parms = create_model()

    print(f'training on {len(train_set)} corpus entries with {args.limit_segments or "all"} segments each')
    save_path, epochs = train_model(model_parms, train_set, dev_set, test_set)
    print(f'Model saved to path: {save_path}')

    fig_ctc, fig_ler = visualize_cost(target_dir, epochs, args)
    fig_ctc.savefig(os.path.join(target_dir, 'ctc_cost.png'), bbox_inches='tight')
    fig_ler.savefig(os.path.join(target_dir, 'ler_cost.png'), bbox_inches='tight')


def create_train_dev_test(args, corpus):
    repeat_sample = None

    if args.id:
        if args.id not in corpus.keys:
            print(f'Error: no entry with id={args.id} found!')
            return exit()

        print(f'training on corpus entry with id={args.id}')
        repeat_sample = corpus[args.id]

    if args.ix:
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


def create_model():
    graph = tf.Graph()
    with graph.as_default():
        # Input sequences: Has size [batch_size, max_step_size, num_features], but the batch_size and max_step_size
        # can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features], name='input')

        # Here we use sparse_placeholder that will generate a SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # Sequence length: 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        # single LSTM cell
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        # The second output is the last state and we will not use that
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
        # optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(cost)
        optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9).minimize(cost)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

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
        'decoded': decoded,
        'log_prob': log_prob
    }


def train_model(model_parms, train_set, dev_set, test_set):
    graph = model_parms['graph']
    cost = model_parms['cost']
    optimizer = model_parms['optimizer']
    ler = model_parms['ler']
    inputs = model_parms['inputs']
    targets = model_parms['targets']
    seq_len = model_parms['seq_len']
    decoded = model_parms['decoded']
    log_prob = model_parms['log_prob']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    train_cost_logger = create_cost_logger(target_dir, 'stats.tsv')
    epoch_logger = create_epoch_logger(target_dir)

    with tf.Session(graph=graph, config=config) as session:
        tf.global_variables_initializer().run()

        curr_epoch = 0

        # sliding window over the LER values of the last 10 epochs
        train_lers = collections.deque(np.repeat((math.inf,), 10), maxlen=10)
        train_lers_means = collections.deque(np.repeat((math.inf,), 10), maxlen=10)

        convergence = False
        # train until convergence
        while not convergence:
            add_distortion = curr_epoch > 0 and args.synthesize
            curr_epoch += 1
            num_samples = 0
            train_cost = train_ler = 0
            start = time.time()

            # iterate over batches for current epoch
            for X, Y, batch_seq_len, ground_truths in generate_batches(train_set, distort_audio=add_distortion):
                feed = {inputs: X, targets: Y, seq_len: batch_seq_len}
                batch_cost, _ = session.run([cost, optimizer], feed)

                batch_len = X.shape[0]
                train_cost += batch_cost * batch_len
                train_ler += session.run(ler, feed_dict=feed) * batch_len

                # Decoding
                d = session.run(decoded[0], feed_dict=feed)
                dense_decoded = tf.sparse_tensor_to_dense(d, default_value=0).eval(session=session)

                for i, prediction_enc in enumerate(dense_decoded):
                    ground_truth = ground_truths[i]
                    prediction = decode(prediction_enc)
                    print_prediction(ground_truth, prediction, 'train-set')
                    log_prediction(epoch_logger, ground_truth, prediction, 'dev_set')

                num_samples += batch_len

            # calculate costs for current epoch
            train_cost /= num_samples
            train_ler /= num_samples
            train_lers.append(train_ler)

            # convergence reached if mean LER-rate is below threshold and mean LER change rate is below 1%
            ler_mean = np.array(train_lers).mean()
            train_lers_means.append(ler_mean)
            ler_diff = np.diff(train_lers_means).mean()
            convergence = ler_mean < LER_CONVERGENCE and abs(ler_diff) < 0.01

            # validate cost with a randomly chosen entry from the dev-set that has been randomly shifted
            X_val, Y_val, val_seq_len, val_ground_truths = random.choice(list(generate_batches(dev_set, True)))
            val_feed = {inputs: X_val, targets: Y_val, seq_len: val_seq_len}
            val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)
            train_cost_logger.write_tabbed([curr_epoch, train_cost, train_ler, val_cost, val_ler])

            val_str = f'=== Epoch {curr_epoch}, train_cost = {train_cost:.3f}, train_ler = {train_ler:.3f}, ' \
                      f'val_cost = {val_cost:.3f}, val_ler = {val_ler:.3f}, time = {time.time() - start:.3f}, ' \
                      f'ler_mean = {ler_mean:.3f}, ler_diff = {ler_diff:.3f} ==='
            print(val_str)
            epoch_logger.write(val_str)

        print(f'convergence reached after {curr_epoch} epochs!')
        saver = tf.train.Saver()
        save_path = saver.save(session, os.path.join(target_dir, 'model.ckpt'))

    return save_path, curr_epoch


def generate_batches(corpus_entries, shift_audio=False, distort_audio=False):
    speech_segments = list(seg for corpus_entry in corpus_entries for seg in corpus_entry.speech_segments_not_numeric)
    l = len(speech_segments)
    for ndx in range(0, l, batch_size):
        batch = []
        for speech_segment in speech_segments[ndx:min(ndx + batch_size, l)]:
            audio = speech_segment.audio  # save original audio signal
            if distort_audio:
                speech_segment.audio = distort(audio, speech_segment.rate, tempo=True)
            if shift_audio:
                speech_segment.audio = shift(audio)  # clip audio before calculating MFCC/spectrogram

            features = speech_segment.mfcc() if args.feature_type == 'mfcc' else speech_segment.spectrogram()
            speech_segment.audio = audio  # restore original audio for next epoch

            batch.append((features, speech_segment.text))

        features, ground_truths = zip(*batch)

        X, batch_seq_len = pad_sequences(features)
        Y = sparse_tuple_from([encode(truth) for truth in ground_truths])

        yield X, Y, batch_seq_len, ground_truths


def create_cost_logger(log_dir, log_file):
    cost_logger = create_file_logger(log_dir, log_file)
    cost_logger.write_tabbed(['epoch', 'cost', 'ler'])
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
