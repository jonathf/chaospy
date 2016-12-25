"""
Methods for performing Gaussian quadrature.

Examples
--------
A quadrature rule where the limits are defined by floats::

    >>> abscissas, weights = chaospy.quad_clenshaw_curtis(
    ...         order=3, lower=0., upper=1.)
    >>> print(abscissas)
    [[ 0.    0.25  0.75  1.  ]]
    >>> print(weights)
    [ 0.11111111  0.38888889  0.19444444  0.11111111]

A quadrature rule in higher dimensions through the main interface::

    >>> lower = [0., 0.]
    >>> upper = [1., 1.]
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...         order=1, domain=(lower, upper), rule="gauss_legendre")
    >>> print(abscissas)
    [[ 0.21132487  0.21132487  0.78867513  0.78867513]
     [ 0.21132487  0.78867513  0.21132487  0.78867513]]
    >>> print(weights)
    [ 0.25  0.25  0.25  0.25]

A quadrature rule where the distribution is the weight function::

    >>> dist = chaospy.Gamma()
    >>> abscissas, weights = chaospy.quad_golub_welsch(
    ...         order=2, dist=dist)
    >>> print(abscissas)
    [[ 0.41577456  2.29428036  6.28994508]]
    >>> print(weights)
    [ 0.71109301  0.27851773  0.01038926]
"""
from .combine import combine
from .interface import generate_quadrature, normalize_weights

from .stieltjes import generate_stieltjes
from .collection import *
from .sparse_grid import sparse_grid
from .generator import rule_generator

import chaospy
