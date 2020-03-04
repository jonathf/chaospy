"""Global configuration."""
import os

import pytest

import numpy
import scipy
import chaospy


@pytest.fixture(autouse=True)
def global_setup(doctest_namespace, monkeypatch):
    """Global configuration setup."""
    doctest_namespace["numpy"] = numpy
    doctest_namespace["scipy"] = scipy
    doctest_namespace["chaospy"] = chaospy

    # fix random seeds:
    numpy.random.seed(1000)

    # set debug mode during testing
    environ = os.environ.copy()
    environ["NUMPOLY_DEBUG"] = True
    monkeypatch.setattr("os.environ", environ)
