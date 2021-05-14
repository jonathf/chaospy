import pytest
from sklearn import linear_model
import numpy
import chaospy

LINEAR_MODELS = {
    "none": None,
    "linear": linear_model.LinearRegression(fit_intercept=False),
    "elastic_net": linear_model.MultiTaskElasticNet(alpha=0.0001, fit_intercept=False),
    "lasso": linear_model.MultiTaskLasso(alpha=0.001, fit_intercept=False),
    "lasso_lars": linear_model.LassoLars(alpha=0.0001, fit_intercept=False),
    "lars": linear_model.Lars(n_nonzero_coefs=10, fit_intercept=False),
    "matching_pursuit": linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=10, fit_intercept=False),
    "ridge": linear_model.Ridge(alpha=0.1, fit_intercept=False),
}


@pytest.fixture
def samples(joint):
    return joint.sample(1000, rule="sobol")


@pytest.fixture
def evaluations(model_solver, samples):
    return numpy.array([model_solver(sample) for sample in samples.T])


@pytest.fixture(params=LINEAR_MODELS)
def linear_model(request, expansion_small, samples, evaluations):
    return chaospy.fit_regression(
        expansion_small, samples, evaluations, model=LINEAR_MODELS[request.param])


def test_regression_mean(linear_model, joint, true_mean):
    assert numpy.allclose(chaospy.E(linear_model, joint), true_mean, rtol=1e-2)


def test_regression_variance(linear_model, joint, true_variance):
    assert numpy.allclose(chaospy.Var(linear_model, joint), true_variance, rtol=3e-1)
