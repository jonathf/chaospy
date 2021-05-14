import chaospy
import numpy
import pytest


@pytest.fixture
def samples(joint):
    return joint.sample(10, rule="sobol")


@pytest.fixture
def evaluations(model_solver, samples):
    return numpy.array([model_solver(sample) for sample in samples.T])


@pytest.fixture
def expansion(samples):
    return chaospy.lagrange_polynomial(samples)


@pytest.fixture
def lagrange_approximation(evaluations, expansion):
    return chaospy.sum(evaluations.T*expansion, axis=-1).T


def test_lagrange_mean(lagrange_approximation, joint, true_mean):
    assert numpy.allclose(chaospy.E(lagrange_approximation, joint), true_mean, rtol=1e-3)


def test_lagrange_variance(lagrange_approximation, joint, true_variance):
    assert numpy.allclose(chaospy.Var(lagrange_approximation, joint), true_variance, rtol=1e-2)
