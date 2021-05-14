"""Testing polynomial related to distributions."""
import chaospy
import numpy
import pytest

DISTRIBUTIONS = {
    "discrete": chaospy.DiscreteUniform(-10, 10),
    "normal": chaospy.Normal(0, 1),
    "uniform": chaospy.Uniform(-1, 1),
    "exponential": chaospy.Exponential(1),
    "gamma": chaospy.Gamma(1),
    "beta": chaospy.Beta(3, 3, lower=-1, upper=1),
    "mvnormal": chaospy.MvNormal([0], [1]),
    "custom": chaospy.UserDistribution(
        cdf=lambda x: (x+1)/2,
        pdf=lambda x: 1/2.,
        lower=lambda: -1,
        upper=lambda: 1,
        ppf=lambda q: 2*q-1,
        mom=lambda k: ((k+1.)%2)/(k+1),
        ttr=lambda k: (0., k*k/(4.*k*k-1)),
    ),
}
BUILDERS = {
    "stieltjes": chaospy.expansion.stieltjes,
    "cholesky": chaospy.expansion.cholesky,
    # "gram_schmidt": chaospy.expansion.gram_schmidt,
}


@pytest.fixture(params=DISTRIBUTIONS)
def distribution(request):
    return DISTRIBUTIONS[request.param]


@pytest.fixture(params=BUILDERS)
def builder(request):
    return BUILDERS[request.param]


@pytest.fixture
def expansion_small(builder, distribution):
    return builder(4, distribution, normed=True)

@pytest.fixture
def expansion_large(builder, distribution):
    return builder(7, distribution, normed=True)


@pytest.fixture
def expansion_approx(builder, distribution):
    def not_implemented(*args, **kwargs):
        raise chaospy.UnsupportedFeature()

    distribution._ttr = not_implemented
    distribution._mom = not_implemented
    return builder(4, distribution, normed=True)


def test_orthogonality_small(expansion_small, distribution):
    outer = chaospy.E(chaospy.outer(expansion_small, expansion_small), distribution)
    assert numpy.allclose(outer, numpy.eye(len(outer)), rtol=1e-8)


def test_orthogonality_large(expansion_large, distribution):
    outer = chaospy.E(chaospy.outer(expansion_large, expansion_large), distribution)
    assert numpy.allclose(outer, numpy.eye(len(outer)), rtol=1e-4)


def test_approx_expansion(expansion_approx, expansion_small, distribution):
    outer1 = chaospy.E(chaospy.outer(expansion_small, expansion_small), distribution)
    outer2 = chaospy.E(chaospy.outer(expansion_approx, expansion_approx), distribution)
    assert numpy.allclose(outer1, outer2, rtol=1e-12)
