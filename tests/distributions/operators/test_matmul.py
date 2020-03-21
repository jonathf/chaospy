from pytest import raises
import numpy

import chaospy

UNIVARIATE = chaospy.Uniform(2, 3)
MULTIVARIATE = chaospy.J(chaospy.Uniform(1, 2), chaospy.Uniform(2, 4))
DEPENDENT1 = chaospy.J(UNIVARIATE, chaospy.Uniform(-2, 1)*UNIVARIATE)
DEPENDENT2 = chaospy.J(UNIVARIATE, UNIVARIATE*chaospy.Uniform(-2, 1))

EPS = 1E-1


def test_dist_matmul_illigals():
    with raises(ValueError):
        UNIVARIATE @ 4
    with raises(ValueError):
        4 @ UNIVARIATE
    with raises(ValueError):
        MULTIVARIATE @ 4
    with raises(ValueError):
        4 @ MULTIVARIATE
    with raises(ValueError):
        UNIVARIATE @ [2, 3]
    with raises(ValueError):
        [2, 3] @ UNIVARIATE
    with raises(ValueError):
        chaospy.Matmul([3, 4], [2, 3])


def test_dist_matmul_lower():
    assert all((MULTIVARIATE @ [2, 3]).lower == [2, 6])
    assert all(([2, 3] @ MULTIVARIATE).lower == [2, 6])
    assert all((MULTIVARIATE @ [[1, 1], [0, 1]]).lower == [1, 3])
    assert all(([[1, 1], [0, 1]] @ MULTIVARIATE).lower == [3, 2])


def test_dist_matmul_upper():
    assert all((MULTIVARIATE @ [2, 3]).upper == [4, 12])
    assert all(([2, 3] @ MULTIVARIATE).upper == [4, 12])
    assert all((MULTIVARIATE @ [[1, 1], [0, 1]]).upper == [2, 6])
    assert all(([[1, 1], [0, 1]] @ MULTIVARIATE).upper == [6, 4])


def test_dist_matmul_forward():
    assert numpy.allclose((MULTIVARIATE @ [2, 3]).fwd([[2, 3, 4], [12, 9, 6]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])
    assert numpy.allclose(([2, 3] @ MULTIVARIATE).fwd([[2, 3, 4], [12, 9, 6]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])
    assert numpy.allclose((MULTIVARIATE @ [[1, 1], [0, 1]]).fwd([[1, 1.5, 2], [6, 4.5, 3]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])
    assert numpy.allclose(([[1, 1], [0, 1]] @ MULTIVARIATE).fwd([[3, 4.5, 6], [4, 3, 2]]),
                          [[0, 0.5, 1], [1, 0.5, 0]])


def test_dist_matmul_inverse():
    assert numpy.allclose((MULTIVARIATE @ [2, 3]).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[2, 3, 4], [12, 9, 6]])
    assert numpy.allclose(([2, 3] @ MULTIVARIATE).inv([[0, 0.5, 1], [1, 0.5, 0]]),
                          [[2, 3, 4], [12, 9, 6]])
    assert numpy.allclose((MULTIVARIATE @ [[1, 1], [0, 1]]).inv([[0, 0.5, 1], [0, 0.5, 1]]),
                          [[1, 1.5, 2], [3, 4.5, 6]])
    assert numpy.allclose(([[1, 1], [0, 1]] @ MULTIVARIATE).inv([[0, 0.5, 1], [0, 0.5, 1]]),
                          [[3, 4.5, 6], [2, 3, 4]])


def test_dist_matmul_density():
    assert numpy.allclose((MULTIVARIATE @ [2, 3]).pdf([[2-EPS, 2+EPS, 4-EPS, 4+EPS],
                                                       [6-EPS, 6+EPS, 12-EPS, 12+EPS]]),
                          [[0, 1/12, 1/12, 0]])
    assert numpy.allclose(([2, 3] @ MULTIVARIATE).pdf([[2-EPS, 2+EPS, 4-EPS, 4+EPS],
                                                       [6-EPS, 6+EPS, 12-EPS, 12+EPS]]),
                          [[0, 1/12, 1/12, 0]])
    assert numpy.allclose((MULTIVARIATE @ [[1, 1], [0, 1]]).pdf([[1, 1, 1, 1, 2, 2, 2, 2],
                                                                 [3-EPS, 3+EPS, 5-EPS, 5+EPS,
                                                                  4-EPS, 4+EPS, 6-EPS, 6+EPS]]),
                          [0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0])
    assert numpy.allclose(([[1, 1], [0, 1]] @ MULTIVARIATE).pdf([[3-EPS, 3+EPS, 5-EPS, 5+EPS,
                                                                  4-EPS, 4+EPS, 6-EPS, 6+EPS],
                                                                 [2, 2, 4, 4, 2, 2, 4, 4]]),
                          [0, 0.25, 0, 0.25, 0.25, 0, 0.25, 0])


def test_dist_matmul_moment():
    pass
    # assert all((MULTIVARIATE @ [2, 3]).mom([[1, 0, 1], [0, 1, 1]]) == [0, 0])
    # assert all(([2, 3] @ MULTIVARIATE).mom([[1, 0, 1], [0, 1, 1]]) == [0, 0])
    # assert all((MULTIVARIATE @ [[1, 1], [0, 1]]).mom([[1, 0, 1], [0, 1, 1]]) == [0, 0])
    # assert all(([[1, 1], [0, 1]] @ MULTIVARIATE).mom([[1, 0, 1], [0, 1, 1]]) == [0, 0])


def test_dist_matmul_recurrence():
    with raises(chaospy.DependencyError):
        _ = (MULTIVARIATE @ [2, 3]).ttr([1, 1])
