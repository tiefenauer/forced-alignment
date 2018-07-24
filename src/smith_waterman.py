import itertools

import numpy as np


def smith_waterman(a, b, match_score=3, gap_cost=2):
    """
    Find b in a
    implementation of Smith-Waterman algorithm according to en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
    :param a: string
    :param b: string
    :return: (start:end) start and end index for local alignment of b with a
    """
    # normalize by making everything uppercase
    # this could be enhanced by removing special characters etc. as in util.string_util.normalize but then the
    # positional information does not match up with the original input strings!
    a, b = a.upper(), b.upper()
    H = matrix(a, b, match_score, gap_cost)
    b_, pos = traceback(H, b)
    return pos, pos + len(b_), b_


def matrix(a, b, match_score=3, gap_cost=2):
    """
    Create scoring matrix H to find b in a by applying the Smith-Waterman algorithm
    :param a: string a (reference string)
    :param b: string b to be aligned with a
    :param match_score: (optional) score to use if a[i] == b[j]
    :param gap_cost: (optional) cost to use for inserts in a or deletions in b
    :return: the scoring matrix H
    """
    H = np.zeros((len(a) + 1, len(b) + 1), np.int)

    for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
        match = H[i - 1, j - 1] + (match_score if a[i - 1] == b[j - 1] else - match_score)
        delete = H[i - 1, j] - gap_cost
        insert = H[i, j - 1] - gap_cost
        H[i, j] = max(match, delete, insert, 0)
    return H


def traceback(H, b, b_='', old_i=0):
    """
    Finds the start position of string b in string a by recursively backtracing a scoring matrix H.
    Note: string a is not needed for this because only the position is searched.
    :param H: scoring matrix (numpy 2D-array)
    :param b: string b to find in string a
    :param b_: (optional) string b_ from previous recursion
    :param old_i: (optional) index of row containing the last maximum value from previous iteration
    :return: (b_, pos)
        b_: string b after applying gaps and deletions
        pos: local alignment (starting position of b in a)
    """
    # flip H to get index of last occurrence of H.max()
    H_flip = np.flip(np.flip(H, 0), 1)
    i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
    i, j = np.subtract(H.shape, (i_ + 1, j_ + 1))  # (i, j) are **last** indexes of H.max()
    # print(H, i,j)
    if H[i, j] == 0:
        return b_, i

    # represent characters that are skipped in string A (i.e. insertions in B) with a dash
    rows_skipped = max(old_i - i - 1,0)
    infix = '-' * rows_skipped

    b_ = b[j - 1] + infix + b_
    return traceback(H[0:i, 0:j], b, b_, i)
