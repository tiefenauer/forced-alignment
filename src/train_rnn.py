# RNN implementation inspired by https://github.com/philipperemy/tensorflow-ctc-speech-recognition
import argparse
import time

import numpy as np
import tensorflow as tf
from os.path import exists

from definitions import CORPUS_TARGET_ROOT
from util.corpus_util import load_corpus
from util.log_util import *
from util.plot_util import visualize_cost
from util.rnn_util import CHAR_TOKENS, decode, DummyCorpus, FileLogger, encode, pad_sequences, sparse_tuple_from

# -------------------------------------------------------------
# Constants, defaults and env-vars
# -------------------------------------------------------------
BATCH_SIZE = 50
NUM_EPOCHS = 10000  # number of epochs to train on
NOW = datetime.now()
MAX_SHIFT = 2000  # maximum number of frames to shift the audio
FEATURE_TYPE = 'mfcc'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# -------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="""Train RNN with CTC cost function for speech recognition""")
parser.add_argument('corpus', type=str, choices=['rl', 'ls'],
                    help='corpus on which to train the RNN (rl=ReadyLingua, ls=LibriSpeech')
parser.add_argument('language', type=str,
                    help='language on which to train the RNN')
parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=BATCH_SIZE,
                    help=f'(optional) number of speech segements to include in one batch (default:{BATCH_SIZE})')
parser.add_argument('-f', '--feature_type', type=str, nargs='?', choices=['mfcc', 'spec'], default='mfcc',
                    help=f'(optional) features to use for training (default: {FEATURE_TYPE})')
parser.add_argument('-id', '--id', type=str, nargs='?',
                    help='(optional) specify ID of single corpus entry on which to train on')
parser.add_argument('-ix', '--ix', type=str, nargs='?',
                    help='(optional) specify index of single corpus entry on which to train on')
parser.add_argument('-t', '--target_root', type=str, nargs='?', default=CORPUS_TARGET_ROOT,
                    help=f'(optional) root directory where results will be written to (default: {CORPUS_TARGET_ROOT})')
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
target_dir = os.path.join(TARGET_ROOT, NOW.strftime('%Y-%m-%d-%H-%M-%S'))

# log_file_path = os.path.join(target_dir, 'train.log')
# print_to_file_and_console(log_file_path)  # comment out to only log to console
# print(f'Results will be written to: {log_file_path}')

# Hyper-parameters
num_features = 161 if args.feature_type == 'spec' else 13
num_classes = len(CHAR_TOKENS) + 2  # 26 lowercase ASCII chars + space + blank = 28 labels
num_hidden = 100
num_layers = 1

# other
batch_size = args.batch_size
validation_interval = 100  # number of entries to process between validation


def main():
    print(create_args_str(args))

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
    save_path = train_model(model_parms, train_set, dev_set, test_set)
    print(f'Model saved to path: {save_path}')

    fig_ctc, fig_ler = visualize_cost(target_dir)
    fig_ctc.savefig(os.path.join(target_dir, 'cost_ctc_sample.png'), bbox_inches='tight')
    fig_ler.savefig(os.path.join(target_dir, 'cost_ler_sample.png'), bbox_inches='tight')


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
        # optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(cost)
        optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9).minimize(cost)

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


def train_model(model_parms, train_set, dev_set, test_set):
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

    train_cost_logger = create_cost_logger(target_dir, 'train_cost.tsv')
    epoch_logger = create_epoch_logger(target_dir)

    with tf.Session(graph=graph, config=config) as session:
        tf.global_variables_initializer().run()

        for curr_epoch in range(args.num_epochs):
            num_samples = 0
            train_cost = train_ler = 0
            start = time.time()

            for X, Y, batch_seq_len, ground_truths in generate_batches(train_set, True):
                feed = {inputs: X, targets: Y, seq_len: batch_seq_len}
                batch_cost, _ = session.run([cost, optimizer], feed)

                train_cost += batch_cost * batch_size
                train_ler += session.run(ler, feed_dict=feed) * batch_size

                # Decoding
                d = session.run(decoded[0], feed_dict=feed)
                dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)

                for i, prediction_enc in enumerate(dense_decoded):
                    ground_truth = ground_truths[i]
                    prediction = decode(prediction_enc)
                    print_prediction(ground_truth, prediction, 'train-set')
                    log_prediction(epoch_logger, ground_truth, prediction, 'dev_set')

                num_samples += batch_size

            train_cost /= num_samples
            train_ler /= num_samples

            train_cost_logger.write_tabbed([curr_epoch, train_cost, train_ler])

            val_str = f'=== Epoch {curr_epoch}, train_cost = {train_cost:.3f}, train_ler = {train_ler:.3f}, ' \
                      f'time = {time.time() - start:.3f} ==='
            print(val_str)
            epoch_logger.write(val_str)

        saver = tf.train.Saver()
        save_path = saver.save(session, os.path.join(target_dir, 'model.ckpt'))

    return save_path


def generate_batches(corpus_entries, shift_audio):
    batch = []
    for speech_segment in (seg for corpus_entry in corpus_entries for seg in corpus_entry.speech_segments_not_numeric):
        audio = speech_segment.audio
        if shift_audio:
            max_shift = int(0.01 * len(speech_segment.audio))
            shift = np.random.randint(low=1, high=max_shift)
            speech_segment.audio = audio[shift:]  # clip audio before calculating MFCC/spectrogram

        batch.append((speech_segment.mfcc(), speech_segment.text))
        speech_segment.audio = audio  # restore original audio

        if len(batch) == batch_size:
            X, ground_truths = zip(*batch)

            X, batch_seq_len = pad_sequences(X)
            Y = sparse_tuple_from([encode(truth) for truth in ground_truths])

            yield X, Y, batch_seq_len, ground_truths

            batch = []


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
