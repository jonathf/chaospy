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

__version__ = "1.1"
__author__ = "Jonathan Feinberg, jonathan@feinberg.no"

import chaospy.bertran
import chaospy.cholesky
import chaospy.descriptives

import chaospy.dist
import chaospy.utils
import chaospy.orthogonal
import chaospy.poly
import chaospy.collocation
import chaospy.quadrature
import chaospy.saltelli

from chaospy.bertran import *
from chaospy.dist import *
from chaospy.utils import *
from chaospy.orthogonal import *
from chaospy.poly import *
from chaospy.collocation import *
from chaospy.quadrature import *
from chaospy.saltelli import *

from chaospy.descriptives import *
