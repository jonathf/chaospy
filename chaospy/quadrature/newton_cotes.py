"""

Example usage
-------------

Generate Newton-Cotes quadrature rules::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> for order in range(5):
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="newton_cotes")
    ...     print(order, abscissas.round(3), weights.round(3))
    0 [[0.5]] [1.]
    1 [[0. 1.]] [0.5 0.5]
    2 [[0.  0.5 1. ]] [0.167 0.667 0.167]
    3 [[0.    0.333 0.667 1.   ]] [0.125 0.375 0.375 0.125]
    4 [[0.   0.25 0.5  0.75 1.  ]] [0.078 0.356 0.133 0.356 0.078]

The first few orders with exponential growth rule::

    >>> for order in range(4):  # doctest: +NORMALIZE_WHITESPACE
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="newton_cotes", growth=True)
    ...     print(order, abscissas.round(3), weights.round(3))
    0 [[0.5]] [1.]
    1 [[0.  0.5 1. ]] [0.167 0.667 0.167]
    2 [[0.   0.25 0.5  0.75 1.  ]] [0.078 0.356 0.133 0.356 0.078]
    3 [[0.    0.125 0.25  0.375 0.5   0.625 0.75  0.875 1.   ]]
       [ 0.035  0.208 -0.033  0.37  -0.16   0.37  -0.033  0.208  0.035]

Applying Smolyak sparse grid on Newton-Cotes::

    >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     2, distribution, rule="newton_cotes",
    ...     growth=True, sparse=True)
    >>> abscissas.round(3)
    array([[0.  , 0.  , 0.  , 0.25, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.75, 1.  ,
            1.  , 1.  ],
           [0.  , 0.5 , 1.  , 0.5 , 0.  , 0.25, 0.5 , 0.75, 1.  , 0.5 , 0.  ,
            0.5 , 1.  ]])
    >>> weights.round(3)
    array([ 0.028,  0.022,  0.028,  0.356,  0.022,  0.356, -0.622,  0.356,
            0.022,  0.356,  0.028,  0.022,  0.028])
"""
from __future__ import division
try:
    from functools import lru_cache
except ImportError:  # pragma: no cover
    from functools32 import lru_cache
import numpy
from scipy import integrate

from .hypercube import hypercube_quadrature


def newton_cotes(order, domain=(0, 1), growth=False, segments=1):
    """
    Generate the abscissas and weights in Newton-Cotes quadrature.

    Newton-Cotes quadrature, are a group of formulas for numerical integration
    based on evaluating the integrand at equally spaced points.

    Args:
        order (int, numpy.ndarray:):
            Quadrature order.
        domain (:func:`chaospy.Distribution`, ;class:`numpy.ndarray`):
            Either distribution or bounding of interval to integrate over.
        growth (bool):
            If True sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples.
        segments (int):
            Split intervals into N subintervals and create a patched
            quadrature based on the segmented quadrature. Can not be lower than
            `order`. If 0 is provided, default to square root of `order`.
            Nested samples only exist when the number of segments are fixed.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            abscissas:
                The quadrature points for where to evaluate the model function
                with ``abscissas.shape == (len(dist), N)`` where ``N`` is the
                number of samples.
            weights:
                The quadrature weights with ``weights.shape == (N,)``.

    Examples:
        >>> abscissas, weights = chaospy.quadrature.newton_cotes(4)
        >>> abscissas.round(4)
        array([[0.  , 0.25, 0.5 , 0.75, 1.  ]])
        >>> weights.round(4)
        array([0.0778, 0.3556, 0.1333, 0.3556, 0.0778])
        >>> abscissas, weights = chaospy.quadrature.newton_cotes(4, segments=2)
        >>> abscissas.round(4)
        array([[0.  , 0.25, 0.5 , 0.75, 1.  ]])
        >>> weights.round(4)
        array([0.0833, 0.3333, 0.1667, 0.3333, 0.0833])

    """
    order = numpy.asarray(order)
    order = numpy.where(growth, numpy.where(order, 2**order, 0), order)
    return hypercube_quadrature(
        _newton_cotes,
        order=order,
        domain=domain,
        segments=segments,
    )


@lru_cache(None)
def _newton_cotes(order):
    """Backend for Newton-Cotes quadrature rule."""
    if order == 0:
        return numpy.full((1, 1), 0.5), numpy.ones(1)
    return numpy.linspace(0, 1, order+1), integrate.newton_cotes(order)[0]/order
