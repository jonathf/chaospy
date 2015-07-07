R"""
Uncertainty Quantifican Toolbox
===============================

This module contains tools for performing uncertainty
quantification of models.

Submodule
---------
bertran         Multi-indexing tools using Bertran's notation
cholesky        Collection of modified Cholesky rutines
collocation     Tools for creating polynomial chaos expansion
descriptives    Statistical descriptive tools
dist            Collection of probability distribution
orthogonal      Orthogonalization toolbox
poly            General creation and manipulation of polynomials
quadrature      Gaussian quadrature toolbox
utils           Supporting function not fitting in anywhere else
"""

__version__ = "1.0"
__author__ = "Jonathan Feinberg, jonathan@feinberg.no"

from cholesky import *
from dist import *
from utils import *
from bertran import *
from descriptives import *
from orthogonal import *
from poly import *
from collocation import *
from quadrature import *
from saltelli import Saltelli
