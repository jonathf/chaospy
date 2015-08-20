import chaospy as cp
import numpy as np
from scipy import special

dim = 3  # <100
samples = 10**2  # <10**6
order = 4 # <20
size = 10**2 # <10**5


class normal(cp.Dist):
    "stripped down normal distribution"

    def _cdf(self, x):
        return special.ndtr(x)

    def _bnd(self):
        return -7.5, 7.5


def test_dist():
    dist = [cp.Normal()]
    for d in xrange(dim-1):
        dist.append(cp.Normal(dist[-1]))
    dist = cp.J(*dist)
    out = dist.sample(samples)
    out = dist.fwd(out)
    out = dist.inv(out)


def test_quasimc():
    dist = [cp.Normal()]
    for d in xrange(dim-1):
        dist.append(cp.Normal(dist[-1]))
    dist = cp.J(*dist)
    dist.sample(samples, "H")
    dist.sample(samples, "M")
    dist.sample(samples, "S")


def test_approx_dist():
    dist = [normal()]
    for d in xrange(dim-1):
        dist.append(normal() + dist[-1])
    dist = cp.J(*dist)
    out = dist.sample(samples)
    out = dist.fwd(out)
    out = dist.inv(out)


def test_orthogonals():
    dist = cp.Iid(cp.Normal(), dim)
    cp.orth_gs(order, dist)
    cp.orth_ttr(order, dist)
    cp.orth_chol(order, dist)


def test_approx_orthogonals():
    dist = cp.Iid(normal(), dim)
    cp.orth_gs(order, dist)
    cp.orth_ttr(order, dist)
    cp.orth_chol(order, dist)


def test_quadrature():
    dist = cp.Iid(cp.Normal(), dim)
    gq = cp.generate_quadrature
    nodes, weights = gq(order, dist, rule="C")
    nodes, weights = gq(order, dist, rule="E")
    nodes, weights = gq(order, dist, rule="G")
    nodes, weights = gq(order, dist, rule="C", sparse=True)
    nodes, weights = gq(order, dist, rule="E", sparse=True)
    nodes, weights = gq(order, dist, rule="G", sparse=True)


def test_approx_quadrature():
    dist = cp.Iid(normal(), dim)
    gq = cp.generate_quadrature
    nodes, weights = gq(order, dist, rule="C")
    nodes, weights = gq(order, dist, rule="E")
    nodes, weights = gq(order, dist, rule="G")
    nodes, weights = gq(order, dist, rule="C", sparse=True)
    nodes, weights = gq(order, dist, rule="E", sparse=True)
    nodes, weights = gq(order, dist, rule="G", sparse=True)


def test_integration():
    dist = cp.Iid(cp.Normal(), dim)
    orth, norms = cp.orth_ttr(order, dist, retall=1)
    gq = cp.generate_quadrature
    nodes, weights = gq(order, dist, rule="C")
    vals = np.empty((len(weights), size))
    cp.fit_quadrature(orth, nodes, weights, vals, norms=norms)


def test_regression():
    dist = cp.Iid(cp.Normal(), dim)
    orth, norms = cp.orth_ttr(order, dist, retall=1)
    data = dist.sample(samples)
    vals = np.empty((samples, size))
    cp.fit_regression(orth, data, vals, "LS")
    cp.fit_regression(orth, data, vals, "T", order=0)
    cp.fit_regression(orth, data, vals, "TC", order=0)


def test_descriptives():
    dist = cp.Iid(cp.Normal(), dim)
    orth = cp.orth_ttr(order, dist)
    cp.E(orth, dist)
    cp.Var(orth, dist)
    cp.Cov(orth, dist)
