# pylint: disable=redefined-outer-name
"""Global configuration."""
import os
import shutil

import pytest


@pytest.fixture(autouse=True)
def configuration(doctest_namespace):
    """Global test configuration."""
    # give access to expected modules in all doctest:
    import numpy
    doctest_namespace["numpy"] = numpy
    import scipy
    doctest_namespace["scipy"] = scipy
    import chaospy
    doctest_namespace["chaospy"] = chaospy
    import numpoly
    doctest_namespace["numpoly"] = numpoly

    # fix random seeds:
    from numpy.random import seed
    seed(1000)
