import chaospy
import numpy
import pytest

QUADRATURE_RULES = {
    "gaussian": {"rule": "gaussian"},
    "sparse1": {"rule": ["genz_keister_16", "clenshaw_curtis"], "sparse": True},
    "sparse2": {"rule": ["genz_keister_18", "fejer_1"], "sparse": True},
    "sparse3": {"rule": ["genz_keister_22", "newton_cotes"], "sparse": True},
    "sparse4": {"rule": ["genz_keister_24", "legendre"], "sparse": True},
    "lobatto": {"rule": "lobatto"},
    "radau": {"rule": "radau"},
}


@pytest.fixture(params=QUADRATURE_RULES)
def nodes_and_weights(joint, request):
    return chaospy.generate_quadrature(4, joint, **QUADRATURE_RULES[request.param])


@pytest.fixture
def nodes(nodes_and_weights):
    return nodes_and_weights[0]


@pytest.fixture
def weights(nodes_and_weights):
    return nodes_and_weights[1]


@pytest.fixture
def evaluations(nodes, model_solver):
    return numpy.array([model_solver(node) for node in nodes.T])


@pytest.fixture
def spectral_approx(expansion_small, nodes, weights, evaluations):
    return chaospy.fit_quadrature(expansion_small, nodes, weights, evaluations)


def test_spectral_mean(spectral_approx, joint, true_mean):
    assert numpy.allclose(chaospy.E(spectral_approx, joint), true_mean)


def test_spectral_variance(spectral_approx, joint, true_variance):
    assert numpy.allclose(chaospy.Var(spectral_approx, joint), true_variance)
