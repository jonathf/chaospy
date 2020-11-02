"""Global configuration."""
import os

import pytest

import numpy
import scipy
import sklearn.linear_model


@pytest.fixture(autouse=True)
def global_setup(doctest_namespace, monkeypatch):
    """Global configuration setup."""
    # set debug mode during testing
    environ = os.environ.copy()
    environ["NUMPOLY_DEBUG"] = "1"
    environ["CHAOSPY_DEBUG"] = "1"
    monkeypatch.setattr("os.environ", environ)

    import chaospy
    doctest_namespace["numpy"] = numpy
    doctest_namespace["scipy"] = scipy
    doctest_namespace["chaospy"] = chaospy
    doctest_namespace["sklearn"] = sklearn

    # fix random seeds:
    numpy.random.seed(1000)
