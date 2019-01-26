"""
Uncertainty Quantification Toolbox
==================================

This module contains tools for performing uncertainty quantification of models.
"""
import logging
import os

from chaospy.version import __version__

import chaospy.bertran
import chaospy.chol
import chaospy.descriptives
import chaospy.distributions
import chaospy.orthogonal
import chaospy.poly
import chaospy.spectral
import chaospy.quad
import chaospy.saltelli
import chaospy.regression

from chaospy.distributions import *
from chaospy.orthogonal import *
from chaospy.poly import *
from chaospy.spectral import *
from chaospy.quad import *
from chaospy.saltelli import *
from chaospy.descriptives import *
from chaospy.regression import *

LOGPATH = os.environ.get("CHAOSPY_LOGPATH", os.devnull)
logging.basicConfig(level=logging.DEBUG, filename=LOGPATH, filemode="w")
streamer = logging.StreamHandler()
streamer.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.addHandler(streamer)
