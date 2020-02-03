"""
Newton-Cotes quadrature, are a group of formulas for numerical integration
based on evaluating the integrand at equally spaced points.

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
import numpy
from scipy.integrate import newton_cotes

from .combine import combine_quadrature


def quad_newton_cotes(order, domain=(0, 1), growth=False):
    """
    Generate the abscissas and weights in Newton-Cotes quadrature.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (chaospy.distributions.baseclass.Dist, numpy.ndarray):
            Either distribution or bounding of interval to integrate over.
        growth (bool):
            If True sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            abscissas:
                The quadrature points for where to evaluate the model function
                with ``abscissas.shape == (len(dist), N)`` where ``N`` is the
                number of samples.
            weights:
                The quadrature weights with ``weights.shape == (N,)``.

    Examples:
        >>> abscissas, weights = quad_newton_cotes(3)
        >>> abscissas.round(4)
        array([[0.    , 0.3333, 0.6667, 1.    ]])
        >>> weights.round(4)
        array([0.375, 1.125, 1.125, 0.375])
    """
    from ..distributions.baseclass import Dist
    if isinstance(domain, Dist):
        abscissas, weights = quad_newton_cotes(
            order, (domain.lower, domain.upper), growth)
        weights *= domain.pdf(abscissas).flatten()
        weights /= numpy.sum(weights)
        return abscissas, weights

    order = numpy.asarray(order, dtype=int).flatten()
    lower, upper = domain
    lower = numpy.asarray(lower).flatten()
    upper = numpy.asarray(upper).flatten()
    dim = max(lower.size, upper.size, order.size)
    order = numpy.ones(dim, dtype=int)*order
    lower = numpy.ones(dim)*lower
    upper = numpy.ones(dim)*upper

    results = [_newton_cotes(*args, growth=growth)
               for args in zip(order, lower, upper)]
    abscissas = [args[0] for args in results]
    weights = [args[1] for args in results]
    return combine_quadrature(abscissas, weights)


def _newton_cotes(order, lower, upper, growth):
    """Backend for Newton-Cotes quadrature rule."""
    if order == 0:
        return numpy.array([0.5*(lower+upper)]), numpy.ones(1)
    order = 2**order if growth else order
    return (
        numpy.linspace(lower, upper, order+1),
        newton_cotes(order)[0]/(upper-lower),
    )
