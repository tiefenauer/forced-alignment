import unittest

from hamcrest import assert_that, is_, empty

from webrtc_comparison import calc_overlap, calc_intersections


class TestWebRTCComparison(unittest.TestCase):

    def test_calc_overlap(self):
        assert_that(calc_overlap((0, 10), (0, 10)), is_(10))
        assert_that(calc_overlap((0, 10), (5, 10)), is_(5))
        assert_that(calc_overlap((0, 10), (6, 15)), is_(4))
        assert_that(calc_overlap((0, 10), (10, 20)), is_(0))

    def test_calc_intersections(self):
        a = [(0, 10), (20, 30)]
        b = [(0, 10), (20, 30)]
        intersections = list(calc_intersections(a, b))
        assert_that(intersections, is_([(0, 10), (20, 30)]))

    def test_calc_intersections_partial(self):
        a = [(0, 10), (20, 30)]
        b = [(5, 15), (17, 25)]
        intersections = list(calc_intersections(a, b))
        assert_that(intersections, is_([(5, 10), (20, 25)]))

    def test_calc_intersections_none(self):
        a = [(0, 10), (20, 30)]
        b = [(10, 15), (30, 40)]
        intersections = list(calc_intersections(a, b))
        assert_that(intersections, is_(empty()))
