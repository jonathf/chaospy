# pylint: disable=redefined-outer-name
"""Global configuration."""
import os
import shutil

import pytest


@pytest.fixture(scope="session")
def workspace_folder(tmpdir_factory):
    """Path to pytest workspace directory."""
    path = str(tmpdir_factory.mktemp("workspace"))
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope="session")
def global_setup(workspace_folder):
    """Global configuration setup."""
    del workspace_folder


@pytest.fixture(autouse=True)
def workspace(global_setup, workspace_folder, doctest_namespace):
    """Folder to work from for each test."""
    del global_setup # to please the pylint Gods.

    # give access to expected modules in all doctest:
    import numpy
    doctest_namespace["numpy"] = numpy
    import chaospy
    doctest_namespace["chaospy"] = chaospy

    # fix random seeds:
    from numpy.random import seed
    seed(1000)

    # change to workspace for the duration of test:
    curdir = os.path.abspath(os.path.curdir)
    os.chdir(workspace_folder)
    yield workspace_folder
    os.chdir(curdir)
