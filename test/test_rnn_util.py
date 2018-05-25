import string
from unittest import TestCase

from hamcrest import assert_that, is_

from util.rnn_utils import tokenize, encode_token, decode_token, CHAR_TOKENS


class TestRNNUtils(TestCase):

    def test_tokenize(self):
        assert_that(tokenize('lorem').tolist(), is_(['l', 'o', 'r', 'e', 'm']))

        assert_that(tokenize('lorem ipsum').tolist(),
                    is_(['l', 'o', 'r', 'e', 'm', '<space>', 'i', 'p', 's', 'u', 'm']))

        assert_that(tokenize('bis 13 grad').tolist(),
                    is_(['b', 'i', 's', '<space>', '<unk>', '<unk>', '<space>', 'g', 'r', 'a', 'd']))

        assert_that(tokenize('nur 2 millionen').tolist(),
                    is_(['n', 'u', 'r', '<space>', '<unk>', '<space>', 'm', 'i', 'l', 'l', 'i', 'o', 'n', 'e', 'n']))

    def test_encode_token(self):
        assert_that(encode_token('<space>'), is_(0))
        assert_that(encode_token('<unk>'), is_(len(CHAR_TOKENS) + 1))
        for i, token in enumerate(string.ascii_lowercase, 1):
            assert_that(encode_token(token), is_(i))

    def test_decode_token(self):
        assert_that(decode_token(0), is_(' '))
        assert_that(decode_token(27), is_('#'))
        for i, char in enumerate(string.ascii_lowercase, 1):
            assert_that(decode_token(i), is_(char))
