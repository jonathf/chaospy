import pytest

import chaospy
from chaospy.recurrence import RECURRENCE_ALGORITHMS

ANALYTICAL_DISTRIBUTIONS = {
    "beta": chaospy.Beta(4, 2),
    "expon": chaospy.Exponential(1),
    "gamma": chaospy.Gamma(2, 2),
    "lognorm": chaospy.LogNormal(-10, 0.1),
    "normal": chaospy.Normal(2, 3),
    "student": chaospy.StudentT(df=25, mu=0.5),
    "uniform": chaospy.Uniform(-1, 2),
}


@pytest.fixture(params=RECURRENCE_ALGORITHMS)
def recurrence_algorithm(request):
    """Parameterization of name of recurrence algorithms."""
    yield request.param


@pytest.fixture(params=ANALYTICAL_DISTRIBUTIONS.keys())
def analytical_distribution(request):
    """Parameterization of distribution with analytical TTR methods."""
    return ANALYTICAL_DISTRIBUTIONS[request.param]
