import numpy
import pytest


def test_monte_carlo_mean(evaluations_large, true_mean):
    assert numpy.allclose(numpy.mean(evaluations_large, axis=0), true_mean, rtol=1e-3)


def test_monte_carlo_variance(evaluations_large, true_variance):
    assert numpy.allclose(numpy.var(evaluations_large, axis=0), true_variance, rtol=2e-2)
