"""Testing polynomial related to distributions"""
import chaospy as cp
import numpoly as nu
import numpy as np


def test_basic_mom():
    dist = cp.Normal(0, 1)
    res = np.array([1, 0, 1, 0, 3])
    assert np.allclose(dist.mom(np.arange(5)), res)


def test_operator_E():
    dist = cp.Normal(0, 1)
    res = np.array([1, 0, 1, 0, 3])
    x = nu.symbols("x")
    poly = x**np.arange(5)
    assert np.allclose(cp.E(poly, dist), res)


def test_orth_ttr():
    dist = cp.Normal(0, 1)
    orth = cp.orth_ttr(5, dist)
    outer = nu.outer(orth, orth)
    Cov1 = cp.E(outer, dist)
    Diatoric = Cov1 - np.diag(np.diag(Cov1))
    assert np.allclose(Diatoric, 0)

    Cov2 = cp.Cov(orth[1:], dist)
    assert np.allclose(Cov1[1:,1:], Cov2)


def test_orth_chol():
    dist = cp.Normal(0, 1)
    orth1 = cp.orth_ttr(5, dist, normed=True)
    orth2 = cp.orth_chol(5, dist, normed=True)
    assert nu.allclose(orth1, orth2)


def test_orth_norms():
    dist = cp.Normal(0, 1)
    orth = cp.orth_ttr(5, dist, normed=True)
    norms = cp.E(orth**2, dist)
    assert np.allclose(norms, 1)
