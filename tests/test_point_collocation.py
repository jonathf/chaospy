import chaospy
import numpy
import pytest


@pytest.fixture
def collocation_model(expansion_small, samples_small, evaluations_small):
    return chaospy.fit_regression(expansion_small, samples_small, evaluations_small)


def test_collocation_mean(collocation_model, joint, true_mean):
    assert numpy.allclose(chaospy.E(collocation_model, joint), true_mean, rtol=1e-6)


def test_regression_variance(collocation_model, joint, true_variance):
    assert numpy.allclose(chaospy.Var(collocation_model, joint), true_variance, rtol=1e-5)
