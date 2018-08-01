import string
from abc import ABC

import numpy as np

SPACE_TOKEN = '<space>'
CHAR_TOKENS = string.ascii_lowercase


def tokenize(text):
    """Splits a text into tokens.
    The text must only contain the lowercase characters a-z and digits. This must be ensured prior to calling this
    method for performance reasons. The tokens are the characters in the text. A special <space> token is added between
    the words. Since numbers are a special case (e.g. '1' and '3' are 'one' and 'three' if pronounced separately, but
    'thirteen' if the text is '13'), digits are mapped to the special '<unk>' token.
    """

    text = text.replace(' ', '  ')
    words = text.split(' ')

    tokens = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in words])
    return tokens


def encode(text):
    tokens = tokenize(text)
    return np.array([encode_token(token) for token in tokens])


def encode_token(token):
    return 0 if token == SPACE_TOKEN else CHAR_TOKENS.index(token) + 1


def decode(tokens):
    return ''.join([decode_token(x) for x in tokens])


def decode_token(ind):
    return ' ' if ind == 0 else CHAR_TOKENS[ind - 1]


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
        :param sequences: a list of lists of type dtype where each element is a sequence
        :param dtype: data type of array
    Returns:
        A tuple (indices, values, shape)
    """
    indices, values = [], []

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


class FileLogger(ABC):
    def __init__(self, file_path):
        self.logfile = open(file_path, 'w')

    def write(self, line):
        self.logfile.writelines(line + '\n')
        self.logfile.flush()

    def write_tabbed(self, elements):
        line = '\t'.join(str(e) for e in elements)
        self.write(line)

    def close(self):
        self.logfile.close()
