import argparse
import os
from os.path import join, exists

from keras.utils import get_custom_objects

from core.ctc_util import clipped_relu
from core.keras_util import load_model
from util.corpus_util import get_corpus
from core.dataset_generator import generate_train_dev_test

parser = argparse.ArgumentParser(
    description="""Evaluate BRNN by using Keras functions and/or making predictions on some samples from test-set""")
parser.add_argument('-m', '--model', type=str, nargs='?', default=None,
                    help='corpus on which to train the RNN (rl=ReadyLingua, ls=LibriSpeech')
parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=5,
                    help=f'(optional) number of speech segments to include in one batch (default: 5)')
args = parser.parse_args()

# set model path here or in args
model_path = r'E:\2018-07-30-10-41-04_BRNN_DS1_rl_en_mfcc'


def main():
    model_path = get_model_path(args.model)
    print(f'evaluating model: {model_path}')

    print('loading model...')
    model = load_model(model_path)
    print('...done!')

    corpus_id, language, feature_type = parse_train_args(model_path)
    print(f'parsed: corpus_id={corpus_id}, language={language}, feature_type={feature_type}')
    corpus = get_corpus(corpus_id, language)
    train_it, val_it, test_it = generate_train_dev_test(corpus, language, feature_type, args.batch_size)
    for batch_inputs, batch_outputs in test_it:
        batch_predictions = model.predict(batch_inputs, batch_size=5)
        print(batch_predictions)


def get_model_path(model=None):
    model_root_path = model or model_path
    if not model_root_path:
        raise ValueError('model path must be supplied either as CLI arg or hardcoded')
    path = join(model_root_path, 'model.h5')
    if not exists(path):
        raise ValueError(f'model not found: {path}')
    return path


def parse_train_args(model_path):
    # parse corpus ID
    corpus_id = 'ls' if '_ls_' in model_path else 'rl'

    # parse language
    lang_start = model_path.index(f'_{corpus_id}_') + 4
    language = model_path[lang_start:lang_start + 2]

    # parse feature type
    feature_start = model_path.rfind('_') + 1
    feature_end = model_path.rfind(os.sep)
    feature_type = model_path[feature_start:feature_end]
    return corpus_id, language, feature_type


def evaluate_model(model, test_it):
    print(f'Evaluating on {len(test_it)} batches ({test_it.n} speech segments)')
    model.evaluate_generator(test_it)

    # for inputs, outputs in test_batches.next_batch():
    #     X_lengths = inputs['input_length']
    #     Y_pred = model.predict(inputs)
    #     res = tf.keras.backend.ctc_decode(Y_pred, X_lengths)
    #     print(f'prediction: {decode(y_pred)}')
    #     print(f'actual: {decode(y)}')


if __name__ == '__main__':
    main()
