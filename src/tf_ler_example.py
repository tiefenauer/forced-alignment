import numpy as np
import tensorflow as tf

from util.rnn_util import encode


def create_sparse_vec(word_list, encode_strings=False):
    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi, x in enumerate(word_list) for yi, y in enumerate(x)]
    if encode_strings:
        values = np.concatenate([encode(word) for word in word_list]).tolist()
    else:
        values = list(''.join(word_list))

    return tf.SparseTensorValue(indices, values, [num_words, 1, 1])


test_string = ['hund']
ref_strings = ['hund', 'hand', 'huno', 'hano', 'hundi', 'handi']

with tf.Session() as sess:
    print('example with unencoded strings')
    labels = create_sparse_vec(test_string * len(ref_strings))
    print(labels)
    prediction = create_sparse_vec(ref_strings)
    print(prediction)
    ler = sess.run(tf.edit_distance(prediction, labels, normalize=True))
    print(ler)

with tf.Session() as sess:
    print('example with encoded strings yield the same result')
    labels = create_sparse_vec(test_string * len(ref_strings), encode_strings=True)
    print(labels)
    prediction = create_sparse_vec(ref_strings, encode_strings=True)
    print(prediction)
    ler = sess.run(tf.edit_distance(prediction, labels, normalize=True))
    print(ler)
