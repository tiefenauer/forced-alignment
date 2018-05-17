from unittest import TestCase

from hamcrest import assert_that, is_

from rnn_utils import tokenize, encode_token


class TestRNNUtils(TestCase):

    def test_split_tokens(self):
        assert_that(tokenize('lorem').tolist(), is_(['l', 'o', 'r', 'e', 'm']))
        assert_that(tokenize('LOREM').tolist(), is_(['l', 'o', 'r', 'e', 'm']))

        assert_that(tokenize('lorem ipsum').tolist(),
                    is_(['l', 'o', 'r', 'e', 'm', '<space>', 'i', 'p', 's', 'u', 'm']))
        assert_that(tokenize('lorem   ipsum').tolist(),
                    is_(['l', 'o', 'r', 'e', 'm', '<space>', 'i', 'p', 's', 'u', 'm']))

    def test_encode_token(self):
        assert_that(encode_token('<space>'), is_(0))
        for i, token in enumerate('abcdefghijklmnopqrstuvwxyzäöü', 1):
            assert_that(encode_token(token), is_(i))
