"""Test nested quadrature rules."""
import pytest
import numpy
import chaospy


NESTED_SCHEMES = "clenshaw_curtis", "fejer_2", "newton_cotes", "grid"
DISTTRIBUTIONS = {
    "univariate": chaospy.Uniform(0, 10),
    "multivariate": chaospy.J(chaospy.Uniform(0, 10), chaospy.Normal(5, 2)),
}


@pytest.fixture(params=NESTED_SCHEMES)
def scheme(request):
    """Parameterize nested quadrature scheme name."""
    return request.param


@pytest.fixture(params=DISTTRIBUTIONS.keys())
def distribution(request):
    """Parameterize distributions."""
    return DISTTRIBUTIONS[request.param]


def test_nestedness_low(scheme, distribution):
    """Ensure nested quadrature does what it is suppose to do at low orders."""
    nodes0, _ = chaospy.generate_quadrature(0, distribution, rule=scheme, growth=True)
    nodes1, _ = chaospy.generate_quadrature(1, distribution, rule=scheme, growth=True)
    nodes2, _ = chaospy.generate_quadrature(2, distribution, rule=scheme, growth=True)
    for node in nodes0.T:
        assert numpy.any(numpy.all(numpy.isclose(node, nodes1.T), axis=-1))
    for node in nodes1.T:
        assert numpy.any(numpy.all(numpy.isclose(node, nodes2.T), axis=-1))


def test_nestedness_high(scheme):
    """Ensure nested quadrature does what it is suppose to do at higher orders."""
    distribution = chaospy.Normal(2, 10)
    [nodes7], _ = chaospy.generate_quadrature(7, distribution, rule=scheme, growth=True)
    [nodes8], _ = chaospy.generate_quadrature(8, distribution, rule=scheme, growth=True)
    for node in nodes7:
        assert numpy.any(numpy.isclose(node, nodes8))
