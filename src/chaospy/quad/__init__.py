"""
Methods for performing Gaussian quadrature.

Examples
--------
A quadrature rule where the limits are defined by floats::

    >>> abscissas, weights = chaospy.quad_clenshaw_curtis(
    ...         order=3, lower=0., upper=1.)
    >>> print(numpy.around(abscissas, 4))
    [[0.   0.25 0.75 1.  ]]
    >>> print(numpy.around(weights, 4))
    [0.1111 0.3889 0.1944 0.1111]

A quadrature rule in higher dimensions through the main interface::

    >>> lower = [0., 0.]
    >>> upper = [1., 1.]
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...         order=1, domain=(lower, upper), rule="gauss_legendre")
    >>> print(numpy.around(abscissas, 4))
    [[0.2113 0.2113 0.7887 0.7887]
     [0.2113 0.7887 0.2113 0.7887]]
    >>> print(numpy.around(weights, 4))
    [0.25 0.25 0.25 0.25]

A quadrature rule where the distribution is the weight function::

    >>> dist = chaospy.Gamma()
    >>> abscissas, weights = chaospy.quad_golub_welsch(
    ...         order=2, dist=dist)
    >>> print(numpy.around(abscissas, 4))
    [[0.4158 2.2943 6.2899]]
    >>> print(numpy.around(weights, 4))
    [0.7111 0.2785 0.0104]
"""
from .combine import combine
from .interface import generate_quadrature, normalize_weights

from .stieltjes import generate_stieltjes
from .collection import *
from .sparse_grid import sparse_grid
from .generator import rule_generator
