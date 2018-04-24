"""Testing basic distributions and their operations
"""
import math
from inspect import isclass

import numpy as np
from scipy import stats
import pytest

import chaospy as cp
from chaospy.distributions import collection


DISTRIBUTIONS = tuple(
    attr for attr in (getattr(collection, name) for name in dir(collection))
    if isclass(attr) and issubclass(attr, cp.Dist)
)

@pytest.fixture(params=DISTRIBUTIONS)
def distribution(request):
    return request.param


def test_dist_add(distribution):
    """Test distribution addition."""
    dist1_e = cp.E(distribution() + 2.0)
    dist2_e = cp.E(2.0 + distribution())
    base_e = cp.E(distribution()) + 2.0
    np.testing.assert_allclose(dist1_e, dist2_e, rtol=1e-05, atol=1e-08)
    np.testing.assert_allclose(dist2_e, base_e, rtol=1e-05, atol=1e-08)


# def test_dist_sub():

#     for name, dist in zip(dist_names, dists):
#         dist1_e = cp.E(dist() - 3.0)
#         dist2_e = cp.E(3.0 - dist())
#         base_e = cp.E(dist()) - 3.0
#         np.testing.assert_allclose(dist1_e, -dist2_e, rtol=1e-05, atol=1e-08)
#         np.testing.assert_allclose(dist2_e, -base_e, rtol=1e-05, atol=1e-08)


# def test_dist_mul():

#     for name, dist in zip(dist_names, dists):
#         dist1_e = cp.E(dist() * 9.0)
#         dist2_e = cp.E(9.0 * dist())
#         base_e = cp.E(dist()) * 9.0
#         np.testing.assert_allclose(dist1_e, dist2_e, rtol=1e-05, atol=1e-08)
#         np.testing.assert_allclose(dist2_e, base_e, rtol=1e-05, atol=1e-08)

#         dist1_e = cp.E(dist() * 0.1)
#         dist2_e = cp.E(0.1 * dist())
#         base_e = cp.E(dist()) * 0.1
#         np.testing.assert_allclose(dist1_e, dist2_e, rtol=1e-05, atol=1e-08)
#         np.testing.assert_allclose(dist2_e, base_e, rtol=1e-05, atol=1e-08)


# def test_weibull_rayleigh():

#     lambda_ = 11
#     rayleigh = cp.Rayleigh(scale=lambda_)
#     r0, r1 = rayleigh.range()
#     x = np.linspace(r0, r1, 300, endpoint=True)

#     weibull = cp.Weibull(shape=2, scale=lambda_*math.sqrt(2))
#     weibull_sp = stats.weibull_min(2, scale=lambda_*math.sqrt(2))

#     np.testing.assert_allclose(weibull_sp.pdf(x), weibull.pdf(x), atol=1e-08)
#     np.testing.assert_allclose(rayleigh.pdf(x), weibull.pdf(x), atol=1e-08)
#     np.testing.assert_allclose(rayleigh.fwd(x), weibull.fwd(x), atol=1e-08)


# def test_compare_scipy_Gamma():

#     shape = 3.0

#     dist_cp = cp.Gamma(shape=shape, scale=1.0, shift=0)
#     r0, r1 = dist_cp.range()
#     x = np.linspace(r0, r1, 300, endpoint=True)

#     pdf_sp = stats.gamma.pdf(x, shape, scale=1.0, loc=0)
#     cdf_sp = stats.gamma.cdf(x, shape, scale=1.0, loc=0)

#     np.testing.assert_allclose(pdf_sp, dist_cp.pdf(x))
#     np.testing.assert_allclose(cdf_sp, dist_cp.cdf(x))


# def test_compare_scipy_Lognormal():

#     shape = 3.0

#     dist_cp = cp.Lognormal(mu=0.0, sigma=shape,  scale=1.0, shift=0)
#     r0, r1 = dist_cp.range()
#     x = np.linspace(r0, r1, 300, endpoint=True)

#     pdf_sp = stats.lognorm.pdf(x, shape, scale=1.0, loc=0)
#     cdf_sp = stats.lognorm.cdf(x, shape, scale=1.0, loc=0)

#     np.testing.assert_allclose(pdf_sp, dist_cp.pdf(x))
#     np.testing.assert_allclose(cdf_sp, dist_cp.cdf(x))


# def test_compare_scipy_Exponweibull():

#     a = 3.0
#     c = 2.0

#     dist_cp = cp.Exponweibull(a=a, c=c,  scale=1.0, shift=0)
#     r0, r1 = dist_cp.range()
#     x = np.linspace(r0, r1, 300, endpoint=True)

#     pdf_sp = stats.exponweib.pdf(x, a, c, scale=1.0, loc=0)
#     cdf_sp = stats.exponweib.cdf(x, a, c, scale=1.0, loc=0)
#     np.testing.assert_allclose(pdf_sp, dist_cp.pdf(x), atol=1e-08)
#     np.testing.assert_allclose(cdf_sp, dist_cp.cdf(x), atol=1e-08)

#     dist_cp = cp.Exponweibull(a=a, c=c,  scale=1.0, shift=1.2)
#     pdf_sp = stats.exponweib.pdf(x, a, c, scale=1.0, loc=1.2)
#     cdf_sp = stats.exponweib.cdf(x, a, c, scale=1.0, loc=1.2)
#     np.testing.assert_allclose(pdf_sp, dist_cp.pdf(x), atol=1e-08)
#     np.testing.assert_allclose(cdf_sp, dist_cp.cdf(x), atol=1e-08)
