import chaospy
import numpy
import pytest


SAMPLING_RULES = {
    "random": {"rule": "random", "seed": 1234},
    "additive_recursion": {"rule": "additive_recursion"},
    "halton": {"rule": "halton"},
    "hammersley": {"rule": "hammersley"},
    "latin_hypercube": {"rule": "latin_hypercube"},
    "korobov": {"rule": "korobov"},
    "sobol": {"rule": "sobol"},
    "antithetic": {"rule": "sobol", "antithetic": True},
}


class TestModel:
    """Test model solver."""

    def __init__(self, coordinates):
        """
        Args:
            coordinates (numpy.ndarray):
                The locations to evaluate at.
        """
        self.coordinates = coordinates

    def __call__(self, parameters):
        """
        Simple ordinary differential equation solver.

        Args:
            parameters (numpy.ndarray):
                Hyper-parameters defining the model initial
                conditions alpha and growth rate beta.
                Assumed to have ``len(parameters) == 2``.

        Returns:
            (numpy.ndarray):
                Solution to the equation.
                Same shape as `self.coordinates`.
        """
        alpha, beta = parameters
        return alpha*numpy.e**-(self.coordinates*beta)


@pytest.fixture(params=SAMPLING_RULES)
def samples_large(joint, request):
    return joint.sample(10000, **SAMPLING_RULES[request.param])


@pytest.fixture
def samples_small(samples_large):
    return samples_large[:1000]


@pytest.fixture
def evaluations_large(model_solver, samples_large):
    return numpy.array([model_solver(sample) for sample in samples_large.T])


@pytest.fixture
def evaluations_small(evaluations_large):
    return evaluations_large[:, :1000]


@pytest.fixture
def coordinates():
    return numpy.linspace(0, 10, 1000)


@pytest.fixture
def joint():
    return chaospy.J(chaospy.Normal(1.5, 0.2), chaospy.Uniform(0.1, 0.2))


@pytest.fixture
def expansion_and_norms(joint):
    return chaospy.generate_expansion(6, joint, retall=True)


@pytest.fixture
def expansion_large(expansion_and_norms):
    return expansion_and_norms[0]


@pytest.fixture
def expansion_small(expansion_and_norms):
    return expansion_and_norms[0][:15]


@pytest.fixture
def norms_large(expansion_and_norms):
    return expansion_and_norms[1]


@pytest.fixture
def norms_small(expansion_and_norms):
    return expansion_and_norms[1][:15]


@pytest.fixture
def model_solver(coordinates):
    return TestModel(coordinates=coordinates)


@pytest.fixture
def true_mean(coordinates):
    t = coordinates[1:]
    return numpy.hstack([1.5, 15*(numpy.e**(-0.1*t)-numpy.e**(-0.2*t))/t])


@pytest.fixture
def true_variance(coordinates, true_mean):
    t = coordinates[1:]
    return numpy.hstack([
            2.29, 11.45*(numpy.e**(-0.2*t)-numpy.e**(-0.4*t))/t])-true_mean**2
