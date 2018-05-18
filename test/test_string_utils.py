from unittest import TestCase

from hamcrest import assert_that, is_

from string_utils import replace_not_alphanumeric, normalize, remove_multi_spaces, create_filename, replace_numeric


class TestStringUtils(TestCase):

    def test_remove_multi_spaces(self):
        assert_that(remove_multi_spaces('foo   bar'), is_('foo bar'))
        assert_that(remove_multi_spaces('  foo   bar  '), is_('foo bar'))
        assert_that(remove_multi_spaces(' foo   bar '), is_('foo bar'))

    def test_replace_non_alphanumeric(self):
        assert_that(replace_not_alphanumeric('a$b€c?d!e. fG'), is_('abcde fG'))

    def test_replace_numeric(self):
        assert_that(replace_numeric('foo 123 bar'), is_('foo ### bar'))

    def test_create_filename(self):
        assert_that(create_filename('a$b€c?d!e. fG'), is_('abcde_fg'))

    def test_normalize(self):
        assert_that(normalize(' Mäßigung!    Please 123  '), is_('massigung please 123'))
