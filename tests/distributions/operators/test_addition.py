"""Tests for the addition operator."""
from pytest import raises

import numpy
import chaospy

UNIVARIATE = chaospy.Uniform(-2, 3)
MULTIVARIATE = chaospy.J(chaospy.Uniform(-1, 2), chaospy.Uniform(2, 4))
DEPENDENT1 = chaospy.J(UNIVARIATE, chaospy.Uniform(-2, 1)+UNIVARIATE)
DEPENDENT2 = chaospy.J(UNIVARIATE, UNIVARIATE+chaospy.Uniform(-2, 1))

EPS = 1E-10


def test_dist_addition_illegals():
    # Too large dim on const:
    with raises(chaospy.UnsupportedFeature):
        _ = MULTIVARIATE+[[1, 1], [0, 1]]
    with raises(chaospy.UnsupportedFeature):
        _ = [[1, 1], [0, 1]]+MULTIVARIATE
    # 2-D object, 1-D var:
    with raises(chaospy.StochasticallyDependentError):
        _ = UNIVARIATE+[1, 1]
    with raises(chaospy.StochasticallyDependentError):
        _ = [1, 1]+UNIVARIATE
    # 0-D object:
    with raises(chaospy.StochasticallyDependentError):
        chaospy.Add(2, 3)


def test_dist_addition_lower():
    assert (UNIVARIATE+4).lower == 2
    assert (4+UNIVARIATE).lower == 2

    assert all((MULTIVARIATE+4).lower == [3, 6])
    assert all((4+MULTIVARIATE).lower == [3, 6])
    assert all((MULTIVARIATE+[2, 3]).lower == [1, 5])
    assert all(([2, 3]+MULTIVARIATE).lower == [1, 5])

    assert all(DEPENDENT1.lower == [-2, -4])
    assert all(DEPENDENT2.lower == [-2, -4])

    assert (UNIVARIATE+UNIVARIATE).lower == -4
    assert all((MULTIVARIATE+MULTIVARIATE).lower == [-2, 4])


def test_dist_addition_upper():
    assert (UNIVARIATE+4).upper == 7
    assert (4+UNIVARIATE).upper == 7

    assert all((MULTIVARIATE+4).upper == [6, 8])
    assert all((4+MULTIVARIATE).upper == [6, 8])
    assert all((MULTIVARIATE+[2, 3]).upper == [4, 7])
    assert all(([2, 3]+MULTIVARIATE).upper == [4, 7])

    assert all(DEPENDENT1.upper == [3, 4])
    assert all(DEPENDENT2.upper == [3, 4])

    assert (UNIVARIATE+UNIVARIATE).upper == 6
    assert all((MULTIVARIATE+MULTIVARIATE).upper == [4, 8])


def test_dist_addition_forward():
    assert numpy.allclose((UNIVARIATE+4).fwd([2, 4.5, 7]), [0, 0.5, 1])
    assert numpy.allclose((4+UNIVARIATE).fwd([2, 4.5, 7]), [0, 0.5, 1])

    assert numpy.allclose((MULTIVARIATE+4).fwd([[3, 4.5, 6], [8, 7, 6]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])
    assert numpy.allclose((4+MULTIVARIATE).fwd([[3, 4.5, 6], [8, 7, 6]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])
    assert numpy.allclose((MULTIVARIATE+[2, 3]).fwd([[1, 2.5, 4], [7, 6, 5]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])
    assert numpy.allclose(([2, 3]+MULTIVARIATE).fwd([[1, 2.5, 4], [7, 6, 5]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])

    assert numpy.allclose(DEPENDENT1.fwd([[-2, -2, -2, 0.5, 0.5, 3, 3, 3],
                                          [-4, -2.5, -1, -1.5, 1.5, 1, 2.5, 4]]),
                          [[0, 0, 0, 0.5, 0.5, 1, 1, 1],
                           [0, 0.5, 1, 0, 1, 0, 0.5, 1]])
    assert numpy.allclose(DEPENDENT2.fwd([[-2, -2, -2, 0.5, 0.5, 3, 3, 3],
                                          [-4, -2.5, -1, -1.5, 1.5, 1, 2.5, 4]]),
                          [[0, 0, 0, 0.5, 0.5, 1, 1, 1],
                           [0, 0.5, 1, 0, 1, 0, 0.5, 1]])


def test_dist_addition_inverse():
    assert numpy.allclose((UNIVARIATE+4).inv([0, 0.5, 1]), [2, 4.5, 7])
    assert numpy.allclose((4+UNIVARIATE).inv([0, 0.5, 1]), [2, 4.5, 7])

    assert numpy.allclose((MULTIVARIATE+4).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[3, 4.5, 6], [8, 7, 6]])
    assert numpy.allclose((4+MULTIVARIATE).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[3, 4.5, 6], [8, 7, 6]])
    assert numpy.allclose((MULTIVARIATE+[2, 3]).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[1, 2.5, 4], [7, 6, 5]])
    assert numpy.allclose(([2, 3]+MULTIVARIATE).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[1, 2.5, 4], [7, 6, 5]])

    assert numpy.allclose(DEPENDENT1.inv([[0, 0, 0, 0.5, 0.5, 1, 1, 1],
                                          [0, 0.5, 1, 0, 1, 0, 0.5, 1]]),
                          [[-2, -2, -2, 0.5, 0.5, 3, 3, 3],
                           [-4, -2.5, -1, -1.5, 1.5, 1, 2.5, 4]])
    assert numpy.allclose(DEPENDENT2.inv([[0, 0, 0, 0.5, 0.5, 1, 1, 1],
                                          [0, 0.5, 1, 0, 1, 0, 0.5, 1]]),
                          [[-2, -2, -2, 0.5, 0.5, 3, 3, 3],
                           [-4, -2.5, -1, -1.5, 1.5, 1, 2.5, 4]])


def test_dist_addition_density():
    assert numpy.allclose((UNIVARIATE+4).pdf([2-EPS, 2+EPS, 7-EPS, 7+EPS]),
                          [0, 1/5., 1/5., 0])
    assert numpy.allclose((4+UNIVARIATE).pdf([2-EPS, 2+EPS, 7-EPS, 7+EPS]),
                          [0, 1/5., 1/5., 0])

    assert numpy.allclose((MULTIVARIATE+4).pdf([[3-EPS, 3+EPS, 6-EPS, 6+EPS],
                                                [8+EPS, 8-EPS, 6+EPS, 6-EPS]]),
                          [0, 1/6., 1/6., 0])
    assert numpy.allclose((4+MULTIVARIATE).pdf([[3-EPS, 3+EPS, 6-EPS, 6+EPS],
                                                [8+EPS, 8-EPS, 6+EPS, 6-EPS]]),
                          [[0, 1/6., 1/6., 0]])
    assert numpy.allclose((MULTIVARIATE+[2, 3]).pdf([[1-EPS, 1+EPS, 4-EPS, 4+EPS],
                                                     [7+EPS, 7-EPS, 5+EPS, 5-EPS]]),
                          [[0, 1/6., 1/6., 0]])
    assert numpy.allclose(([2, 3]+MULTIVARIATE).pdf([[1-EPS, 1+EPS, 4-EPS, 4+EPS],
                                                     [7+EPS, 7-EPS, 5+EPS, 5-EPS]]),
                          [[0, 1/6., 1/6., 0]])

    assert numpy.allclose(DEPENDENT1.pdf([[-2, -2, -2, -2, 3, 3, 3, 3],
                                          [-4-EPS, -4+EPS, -1-EPS, -1+EPS, 1-EPS, 1+EPS, 4-EPS, 4+EPS]]),
                          [0, 1/15., 1/15., 0, 0, 1/15., 1/15., 0])
    assert numpy.allclose(DEPENDENT2.pdf([[-2, -2, -2, -2, 3, 3, 3, 3],
                                          [-4-EPS, -4+EPS, -1-EPS, -1+EPS, 1-EPS, 1+EPS, 4-EPS, 4+EPS]]),
                          [0, 1/15., 1/15., 0, 0, 1/15., 1/15., 0])


def test_dist_addition_moment():
    assert (UNIVARIATE+4).mom(1) == 4.5
    assert (4+UNIVARIATE).mom(1) == 4.5
    assert all((MULTIVARIATE+4).mom([[1, 0, 1], [0, 1, 1]]) == [4.5, 7.0, 31.5])
    assert all((4+MULTIVARIATE).mom([[1, 0, 1], [0, 1, 1]]) == [4.5, 7.0, 31.5])


def test_dist_addition_recurrence():
    assert numpy.allclose((UNIVARIATE+4).ttr(1), [4.5, 25/12.])
    assert numpy.allclose((4+UNIVARIATE).ttr(1), [4.5, 25/12.])
    assert numpy.allclose((MULTIVARIATE+4).ttr([1, 0]), [[4.5, 7], [0.75, 0]])
    assert numpy.allclose((4+MULTIVARIATE).ttr([1, 0]), [[4.5, 7], [0.75, 0]])


def test_dist_addition_wrappers():
    dists = [
        chaospy.J(chaospy.Normal(2), chaospy.Normal(2), chaospy.Normal(3)),  # ShiftScale
        chaospy.J(chaospy.Uniform(1, 3), chaospy.Uniform(1, 3), chaospy.Uniform(1, 5)),  # LowerUpper
        chaospy.MvNormal([2, 2, 3], numpy.eye(3)),  # MeanCovariance
    ]
    for dist in dists:
        joint = chaospy.Add([1, 1, 3], dist)

        assert numpy.allclose(joint.inv([0.5, 0.5, 0.5]), [3, 3, 6])
        assert numpy.allclose(joint.fwd([3, 3, 6]), [0.5, 0.5, 0.5])
        density = joint.pdf([2, 2, 2], decompose=True)
        assert numpy.isclose(density[0], density[1]), (dist, density)
        assert not numpy.isclose(density[0], density[2]), dist
        assert numpy.isclose(joint[0].inv([0.5]), 3)
        assert numpy.isclose(joint[0].fwd([3]), 0.5)
