"""
Uncertainty Quantification Toolbox
==================================

This module contains tools for performing uncertainty quantification of models.
"""
from functools import wraps
from contextlib import contextmanager
import logging
import os

import numpoly
from numpoly import *

import chaospy.bertran
import chaospy.chol
import chaospy.descriptives
import chaospy.distributions
import chaospy.orthogonal
import chaospy.spectral
import chaospy.quadrature
import chaospy.saltelli
import chaospy.regression

from chaospy.distributions import *
from chaospy.orthogonal import *
from chaospy.spectral import *
from chaospy.quadrature import *
from chaospy.saltelli import *
from chaospy.descriptives import *
from chaospy.regression import *
from chaospy.external import *

LOGPATH = os.environ.get("CHAOSPY_LOGPATH", os.devnull)
logging.basicConfig(level=logging.DEBUG, filename=LOGPATH, filemode="w")
streamer = logging.StreamHandler()
streamer.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.addHandler(streamer)


@contextmanager
def temporary_set_variable(module, **kwargs):
    """Temporary rename variables in object."""
    current_state = {getattr(module, key) for key in kwargs}
    for key, value in kwargs.items():
        setattr(module, key, value)
    yield	
    for key, value in kwargs.items():
        setattr(module, key, value)


def constrain_namespace(function):
    """Change global scope for indeterminant variables to 'q\d+' format."""
    @wraps(function)
    def wrapper(*args, **kwargs):
        """Function wrapper."""
        with temporary_set_variable(
            module=numpoly.baseclass,
            INDETERMINANT_REGEX=r"q\d+",
            INDETERMINANT_DEFAULT=r"q",
            INDETERMINANT_DEFAULT_INDEX=True,
        ):
            return function(*args, **kwargs)

for name in dir(numpoly):
    if not name.startswith("_") and name in vars():
        vars()[name] = constrain_namespace(vars()[name])
