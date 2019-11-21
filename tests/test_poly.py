"""Testing basic polynomial operators
"""
import chaospy as cp
import numpy as np


def test_poly_variable():
    XYZ = cp.variable(3)
    assert len(XYZ) == 3


def test_poly_representation():
    XYZ = cp.variable(3)
    assert str(XYZ) == "[q0 q1 q2]"


def test_poly_composit():
    X, Y, Z = cp.variable(3)
    ZYX = cp.polynomial([Z, Y, X])
    assert str(ZYX) == "[q2 q1 q0]"


def test_poly_matrix():
    XYZ = cp.variable(3)
    XYYZ = cp.polynomial([XYZ[:2], XYZ[1:]])
    assert str(XYYZ) == "[[q0 q1]\n [q1 q2]]"
    assert XYYZ.shape == (2, 2)


def test_poly_basis():
    basis = cp.basis(1, 2, 2, sort="GR")
    assert str(basis) == "[q0 q1 q0**2 q0*q1 q1**2]"


def test_poly_dimredux():
    basis = cp.basis(1, 2, 2, sort="GR")
    X = cp.variable()
    assert basis[0] == X


def test_poly_power():
    X, Y = cp.variable(2)
    out = X**(1, 0, 2, 1, 0)*Y**(0, 1, 0, 1, 2)
    assert str(out) == "[q0 q1 q0**2 q0*q1 q1**2]"


def test_poly_evals():
    xy = x, y = cp.variable(2)
    assert xy(0.5)[0] == 0.5
    assert xy(0.5)[1] == y


def test_poly_indirect_evals():
    basis = cp.basis(1, 2, 2, sort="GR")
    x, y = cp.variable(2)
    assert basis(q1=3)[1] == 3
    assert basis(q1=3)[3] == 3*x


def test_poly_eval_array():
    xy = cp.variable(2)
    out = xy([1, 2], 3)
    ref = np.array([[1, 2], [3, 3]])
    assert np.allclose(out, ref)


def test_poly_substitute():
    x, y = xy = cp.variable(2)
    assert xy(q1=x)[1] == x


def test_poly_substitute_2():
    x, y = xy = cp.variable(2)
    assert xy(x*y+1)[0] == x*y+1


def test_poly_arithmatic():
    xyz = x, y, z = cp.variable(3)
    assert xyz[0]*.5+y-x/2. == y


def test_poly_linearcomb():

    x, y = xy = cp.variable(2)
    mul1 = xy * np.eye(2)
    mul2 = cp.polynomial([[x, 0], [0, y]])
    assert np.all(mul1 == mul2)
