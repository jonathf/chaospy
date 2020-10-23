"""Tests for the multiplication operator."""
from pytest import raises
import numpy

import chaospy

UNIVARIATE = chaospy.Uniform(-2, 3)
MULTIVARIATE = chaospy.J(chaospy.Uniform(-1, 2), chaospy.Uniform(2, 4))
DEPENDENT1 = chaospy.J(UNIVARIATE, chaospy.Uniform(-2, 1)*UNIVARIATE)
DEPENDENT2 = chaospy.J(UNIVARIATE, UNIVARIATE*chaospy.Uniform(-2, 1))

EPS = 1E-10


def test_dist_multiply_illigals():
    # Too large dim on const:
    with raises(chaospy.UnsupportedFeature):
        _ = MULTIVARIATE*[[1, 1], [0, 1]]
    with raises(chaospy.UnsupportedFeature):
        _ = [[1, 1], [0, 1]]*MULTIVARIATE
    # 2-D object, 1-D var:
    with raises(chaospy.StochasticallyDependentError):
        _ = UNIVARIATE*[1, 1]
    with raises(chaospy.StochasticallyDependentError):
        _ = [1, 1]*UNIVARIATE
    # 0-D object:
    with raises(chaospy.StochasticallyDependentError):
        chaospy.Multiply(2, 3)


def test_dist_multiply_lower():
    assert (UNIVARIATE*4).lower == -8
    assert (4*UNIVARIATE).lower == -8

    assert all((MULTIVARIATE*4).lower == [-4, 8])
    assert all((4*MULTIVARIATE).lower == [-4, 8])
    assert all((MULTIVARIATE*[2, 3]).lower == [-2, 6])
    assert all(([2, 3]*MULTIVARIATE).lower == [-2, 6])

    assert all(DEPENDENT1.lower == [-2, -6])
    assert all(DEPENDENT2.lower == [-2, -6])

    assert (UNIVARIATE*UNIVARIATE).lower == -6
    assert all((MULTIVARIATE*MULTIVARIATE).lower == [-2, 4])


def test_dist_multiply_upper():
    assert (UNIVARIATE*4).upper == 12
    assert (4*UNIVARIATE).upper == 12

    assert all((MULTIVARIATE*4).upper == [8, 16])
    assert all((4*MULTIVARIATE).upper == [8, 16])
    assert all((MULTIVARIATE*[2, 3]).upper == [4, 12])
    assert all(([2, 3]*MULTIVARIATE).upper == [4, 12])

    assert all(DEPENDENT1.upper == [3, 4])
    assert all(DEPENDENT2.upper == [3, 4])

    assert (UNIVARIATE*UNIVARIATE).upper == 9
    assert all((MULTIVARIATE*MULTIVARIATE).upper == [4, 16])


def test_dist_multiply_forward():
    assert numpy.allclose((UNIVARIATE*4).fwd([-8, 2, 12]), [0, 0.5, 1])
    assert numpy.allclose((4*UNIVARIATE).fwd([-8, 2, 12]), [0, 0.5, 1])

    assert numpy.allclose((MULTIVARIATE*4).fwd([[-4, 2, 8], [16, 12, 8]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])
    assert numpy.allclose((4*MULTIVARIATE).fwd([[-4, 2, 8], [16, 12, 8]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])
    assert numpy.allclose((MULTIVARIATE*[2, 3]).fwd([[-2, 1, 4], [12, 9, 6]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])
    assert numpy.allclose(([2, 3]*MULTIVARIATE).fwd([[-2, 1, 4], [12, 9, 6]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])

    assert numpy.allclose(DEPENDENT1.fwd([[-2, -2, -2, 0.5, 0.5, 3, 3, 3],
                                          [-2, 1, 4, -1, 0.5, -6, -1.5, 3]]),
                          [[0, 0, 0, 0.5, 0.5, 1, 1, 1],
                           [0, 0.5, 1, 0, 1, 0, 0.5, 1]])
    assert numpy.allclose(DEPENDENT2.fwd([[-2, -2, -2, 0.5, 0.5, 3, 3, 3],
                                          [-2, 1, 4, -1, 0.5, -6, -1.5, 3]]),
                          [[0, 0, 0, 0.5, 0.5, 1, 1, 1],
                           [0, 0.5, 1, 0, 1, 0, 0.5, 1]])


def test_dist_multiply_inverse():
    assert numpy.allclose((UNIVARIATE*4).inv([0, 0.5, 1]), [-8, 2, 12])
    assert numpy.allclose((4*UNIVARIATE).inv([0, 0.5, 1]), [-8, 2, 12])

    assert numpy.allclose((MULTIVARIATE*4).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[-4, 2, 8], [16, 12, 8]])
    assert numpy.allclose((4*MULTIVARIATE).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[-4, 2, 8], [16, 12, 8]])
    assert numpy.allclose((MULTIVARIATE*[2, 3]).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[-2, 1, 4], [12, 9, 6]])
    assert numpy.allclose(([2, 3]*MULTIVARIATE).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[-2, 1, 4], [12, 9, 6]])

    assert numpy.allclose(DEPENDENT1.inv([[0, 0, 0, 0.5, 0.5, 1, 1, 1],
                                          [0, 0.5, 1, 0, 1, 0, 0.5, 1]]),
                          [[-2, -2, -2, 0.5, 0.5, 3, 3, 3],
                           [-2, 1, 4, -1, 0.5, -6, -1.5, 3]])
    assert numpy.allclose(DEPENDENT2.inv([[0, 0, 0, 0.5, 0.5, 1, 1, 1],
                                          [0, 0.5, 1, 0, 1, 0, 0.5, 1]]),
                          [[-2, -2, -2, 0.5, 0.5, 3, 3, 3],
                           [-2, 1, 4, -1, 0.5, -6, -1.5, 3]])


def test_dist_multiply_density():
    assert numpy.allclose((UNIVARIATE*4).pdf([-8-EPS, -8+EPS, 12-EPS, 12+EPS]),
                          [0, 1/20., 1/20., 0])
    assert numpy.allclose((4*UNIVARIATE).pdf([-8-EPS, -8+EPS, 12-EPS, 12+EPS]),
                          [0, 1/20., 1/20., 0])

    assert numpy.allclose((MULTIVARIATE*4).pdf([[-4-EPS, -4+EPS, 8-EPS, 8+EPS],
                                                [16+EPS, 16-EPS, 8+EPS, 8-EPS]]),
                          [[0, 1/96., 1/96., 0]])
    assert numpy.allclose((4*MULTIVARIATE).pdf([[-4-EPS, -4+EPS, 8-EPS, 8+EPS],
                                                [16+EPS, 16-EPS, 8+EPS, 8-EPS]]),
                          [[0, 1/96., 1/96., 0]])
    assert numpy.allclose((MULTIVARIATE*[2, 3]).pdf([[-2-EPS, -2+EPS, 4-EPS, 4+EPS],
                                                     [12+EPS, 12-EPS, 6+EPS, 6-EPS]]),
                          [[0, 1/36., 1/36., 0]])
    assert numpy.allclose(([2, 3]*MULTIVARIATE).pdf([[-2-EPS, -2+EPS, 4-EPS, 4+EPS],
                                                     [12+EPS, 12-EPS, 6+EPS, 6-EPS]]),
                          [[0, 1/36., 1/36., 0]])

    assert numpy.allclose(DEPENDENT1.pdf([[-2, -2, -2, -2, 3, 3, 3, 3],
                                          [-2-EPS, -2+EPS, 4-EPS, 4+EPS, -6-EPS, -6+EPS, 3-EPS, 3+EPS]]),
                          [0, 1/30., 1/30., 0, 0, 1/45., 1/45., 0])
    assert numpy.allclose(DEPENDENT2.pdf([[-2, -2, -2, -2, 3, 3, 3, 3],
                                          [-2-EPS, -2+EPS, 4-EPS, 4+EPS, -6-EPS, -6+EPS, 3-EPS, 3+EPS]]),
                          [0, 1/30., 1/30., 0, 0, 1/45., 1/45., 0])


def test_dist_multiply_moment():
    assert (UNIVARIATE*4).mom(1) == 2
    assert (4*UNIVARIATE).mom(1) == 2

    assert all((MULTIVARIATE*4).mom([[1, 0, 1], [0, 1, 1]]) == [2, 12, 24])
    assert all((4*MULTIVARIATE).mom([[1, 0, 1], [0, 1, 1]]) == [2, 12, 24])


def test_dist_multiply_recurrence():
    assert numpy.allclose((UNIVARIATE*4).ttr(1), [2, 100/3.])
    assert numpy.allclose((4*UNIVARIATE).ttr(1), [2, 100/3.])

    assert numpy.allclose((MULTIVARIATE*4).ttr([1, 0]), [[2, 12], [12, 0]])
    assert numpy.allclose((4*MULTIVARIATE).ttr([1, 0]), [[2, 12], [12, 0]])
