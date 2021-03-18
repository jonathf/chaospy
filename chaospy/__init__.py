"""
Uncertainty Quantification Toolbox
==================================

This module contains tools for performing uncertainty quantification of models.
"""
import logging
import os
import pkg_resources

from numpoly import *

import chaospy.descriptives
import chaospy.distributions
import chaospy.expansion
import chaospy.spectral
import chaospy.quadrature
import chaospy.saltelli
import chaospy.regression
import chaospy.recurrence

from chaospy.distributions import *
from chaospy.expansion import *
from chaospy.spectral import *
from chaospy.quadrature import *
from chaospy.saltelli import *
from chaospy.descriptives import *
from chaospy.regression import *
from chaospy.external import *
from chaospy.recurrence import *

try:
    __version__ = pkg_resources.get_distribution("chaospy").version
except pkg_resources.DistributionNotFound:  # pragma: no cover
    __version__ = None


def configure_logging():
    """Configure logging for Chaospy."""
    logpath = os.environ.get("CHAOSPY_LOGPATH", os.devnull)
    logging.basicConfig(level=logging.DEBUG, filename=logpath, filemode="w")
    streamer = logging.StreamHandler()
    loglevel = logging.DEBUG if os.environ.get("CHAOSPY_DEBUG", "") == "1" else logging.WARNING
    streamer.setLevel(loglevel)

    logger = logging.getLogger("chaospy")
    logger.addHandler(streamer)
    logger = logging.getLogger("numpoly")
    logger.addHandler(streamer)

configure_logging()


class StochasticallyDependentError(ValueError):
    """Error related to stochastically dependent variables."""


class UnsupportedFeature(NotImplementedError):
    """Error when dependencies are not correctly handled."""
