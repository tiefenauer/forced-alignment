from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
from python_speech_features import mfcc

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
AE_INDEX = ord('z') + 1 - FIRST_INDEX
OE_INDEX = AE_INDEX + 1
UE_INDEX = OE_INDEX + 1

non_alphanumeric_pattern = re.compile('[^a-zA-Zäöü ]+')


def create_x_y(audio, rate, text):
    # create x from MFCC features
    inputs = mfcc(audio, samplerate=rate)
    train_inputs = np.asarray(inputs[np.newaxis, :])
    x = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    # create y from encoded text
    tokens = tokenize(text)
    targets = encode(tokens)
    y = sparse_tuple_from([targets])

    return x, y


def tokenize(text):
    """Splits a text into tokens. The tokens are the words in the text and a special <space> token which is
    added between the words. The text must only contain the following characters: [a-zA-Zäöü ]. This must be done
    prior to calling this method (pre-processed for performance reasons)"""

    text = text.replace(' ', '  ')
    words = text.split(' ')

    tokens = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in words])
    return tokens


def encode(tokens):
    return [encode_token(token) for token in tokens]


def encode_token(token):
    if token == SPACE_TOKEN:
        return SPACE_INDEX
    if token == 'ä':
        return AE_INDEX
    if token == 'ö':
        return OE_INDEX
    if token == 'ü':
        return UE_INDEX
    return ord(token) - FIRST_INDEX


def decode_token(ind):
    return ' abcdefghijklmnopqrstuvwxyzäöü_'[ind]


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths
