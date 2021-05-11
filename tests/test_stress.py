import chaospy as cp
import numpy as np
from scipy import special

dim = 3  # <100
samples = 20  # <10**6
order = 2 # <20
size = 20 # <10**5


class normal(cp.SimpleDistribution):
    """stripped down normal distribution."""
    def _cdf(self, x):
        return special.ndtr(x)
    def _lower(self):
        return -7.5
    def _upper(self):
        return 7.5


def test_dist():
    dist = [cp.Normal()]
    for d in range(dim-1):
        dist.append(cp.Normal(dist[-1]))
    dist = cp.J(*dist)
    out = dist.sample(samples)
    out = dist.fwd(out)
    out = dist.inv(out)


def test_quasimc():
    dist = [cp.Normal()]
    for d in range(dim-1):
        dist.append(cp.Normal(dist[-1]))
    dist = cp.J(*dist)
    dist.sample(samples, "H")
    dist.sample(samples, "M")
    dist.sample(samples, "S")


# def test_approx_dist():
#     dist = [normal()]
#     for d in range(dim-1):
#         dist.append(normal() + dist[-1])
#     dist = cp.J(*dist)
#     out = dist.sample(samples)
#     out = dist.fwd(out)
#     out = dist.inv(out)


def test_orthogonals():
    dist = cp.Iid(cp.Normal(), dim)
    cp.expansion.gram_schmidt(order, dist)
    cp.expansion.stieltjes(order, dist)
    cp.expansion.cholesky(order, dist)


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
    nodes, weights = cp.generate_quadrature(order, dist, rule="C")


def test_integration():
    dist = cp.Iid(cp.Normal(), dim)
    orth, norms = cp.expansion.stieltjes(order, dist, retall=1)
    gq = cp.generate_quadrature
    nodes, weights = gq(order, dist, rule="C")
    vals = np.zeros((len(weights), size))
    cp.fit_quadrature(orth, nodes, weights, vals, norms=norms)


def test_regression():
    dist = cp.Iid(cp.Normal(), dim)
    orth, norms = cp.expansion.stieltjes(order, dist, retall=1)
    data = dist.sample(samples)
    vals = np.zeros((samples, size))
    cp.fit_regression(orth, data, vals)


def test_descriptives():
    dist = cp.Iid(cp.Normal(), dim)
    orth = cp.expansion.stieltjes(order, dist)
    cp.E(orth, dist)
    cp.Var(orth, dist)
    cp.Cov(orth, dist)
