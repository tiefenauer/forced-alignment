import unittest

from hamcrest import assert_that, is_

from smith_waterman import matrix, traceback, smith_waterman


class SmithWatermanTest(unittest.TestCase):

    def test_matrix(self):
        print(matrix('ISSP', 'MISSISSIPPI', match_score=2, gap_cost=1))
        print()
        print(matrix('TCCG', 'ACGA'))
        print()
        print(matrix('AATCG', 'AACG'))
        print()
        print(matrix('GGTTGACTA', 'TGTTACGG'))

    def test_traceback(self):
        a, b = 'ggttgacta', 'tgttacgg'
        H = matrix(a, b)
        b_, pos = traceback(H, b)
        assert_that(pos, is_(1))
        assert_that(b_, is_('gtt-ac'))

        a, b = 'mississippi', 'issp'
        H = matrix(a, b)
        b_, pos = traceback(H, b)
        assert_that(pos, is_(4))
        assert_that(b_, is_('iss-p'))

    def test_smith_waterman(self):
        a, b = 'GGTTGACTA', 'TGTTACGG'
        start, end, b_ = smith_waterman(a, b)
        # print(a[start:end])
        assert_that(start, is_(1))
        assert_that(end, is_(7))
        assert_that(b_, is_('gtt-ac'))

        a, b = 'Mississippi', 'issp'
        start, end, b_ = smith_waterman(a, b)
        # print(a[start:end])
        assert_that(start, is_(4))
        assert_that(end, is_(9))
        assert_that(b_, is_('iss-p'))
