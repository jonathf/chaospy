"""Testing polynomial related to distributions
"""
import chaospy as cp
import numpy as np


def test_basic_mom():
    dist = cp.Normal(0, 1)
    res = np.array([1, 0, 1, 0, 3])
    assert np.allclose(dist.mom(np.arange(5)), res)


def test_operator_E():
    dist = cp.Normal(0, 1)
    res = np.array([1, 0, 1, 0, 3])
    x = cp.variable()
    poly = x**np.arange(5)
    assert np.allclose(cp.E(poly, dist), res)


def test_expansion_stieltjes():
    dist = cp.Normal(0, 1)
    orth = cp.expansion.stieltjes(5, dist)
    outer = cp.outer(orth, orth)
    Cov1 = cp.E(outer, dist)
    Diatoric = Cov1 - np.diag(np.diag(Cov1))
    assert np.allclose(Diatoric, 0)

    Cov2 = cp.Cov(orth[1:], dist)
    assert np.allclose(Cov1[1:,1:], Cov2)


def test_expansion_cholesky():
    dist = cp.Normal(0, 1)
    orth1 = cp.expansion.cholesky(5, dist, normed=True)
    orth2 = cp.expansion.cholesky(5, dist, normed=True)
    eps = cp.sum((orth1-orth2)**2)
    assert np.allclose(eps(np.linspace(-100, 100, 5)), 0)


def test_expansion_stieltjes_norms():
    dist = cp.Normal(0, 1)
    orth = cp.expansion.stieltjes(5, dist, normed=True)
    norms = cp.E(orth**2, dist)
    assert np.allclose(norms, 1)
