"""
Newton-Cotes quadrature, are a group of formulas for numerical integration
based on evaluating the integrand at equally spaced points.

Example usage
-------------

Generate Newton-Cotes quadrature rules::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> for order in range(5):
    ...     X, W = chaospy.generate_quadrature(order, distribution, rule="N")
    ...     print("{} {} {}".format(
    ...         order, numpy.around(X, 3), numpy.around(W, 3)))
    0 [[0.5]] [1.]
    1 [[0. 1.]] [0.5 0.5]
    2 [[0.  0.5 1. ]] [0.333 1.333 0.333]
    3 [[0.    0.333 0.667 1.   ]] [0.375 1.125 1.125 0.375]
    4 [[0.   0.25 0.5  0.75 1.  ]] [0.311 1.422 0.533 1.422 0.311]

The first few orders with exponential growth rule::

    >>> for order in range(4):  # doctest: +NORMALIZE_WHITESPACE
    ...     X, W = chaospy.generate_quadrature(
    ...         order, distribution, rule="N", growth=True)
    ...     print("{} {} {}".format(
    ...         order, numpy.around(X, 3), numpy.around(W, 3)))
    0 [[0.5]] [1.]
    1 [[0.  0.5 1. ]] [0.333 1.333 0.333]
    2 [[0.   0.25 0.5  0.75 1.  ]] [0.311 1.422 0.533 1.422 0.311]
    3 [[0.    0.125 0.25  0.375 0.5   0.625 0.75  0.875 1.   ]]
       [ 0.279  1.662 -0.262  2.962 -1.281  2.962 -0.262  1.662  0.279]

Applying Smolyak sparse grid on Newton-Cotes::

    >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    >>> X, W = chaospy.generate_quadrature(
    ...     2, distribution, rule="N", growth=True, sparse=True)
    >>> print(numpy.around(X, 2))
    [[0.   0.5  1.   0.5  0.   0.25 0.5  0.75 1.   0.5  0.   0.5  1.  ]
     [0.   0.   0.   0.25 0.5  0.5  0.5  0.5  0.5  0.75 1.   1.   1.  ]]
    >>> print(numpy.around(W, 3))
    [0.111 0.422 0.111 1.422 0.422 1.422 0.178 1.422 0.422 1.422 0.111 0.422
     0.111]
"""
import numpy
from scipy.integrate import newton_cotes

from ..combine import combine


def quad_newton_cotes(order, lower=0, upper=1, growth=False):
    """
    Generate the abscissas and weights in Newton-Cotes quadrature.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        lower (int, numpy.ndarray):
            Lower bounds of interval to integrate over.
        upper (int, numpy.ndarray):
            Upper bounds of interval to integrate over.

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
        >>> print(numpy.around(abscissas, 4))
        [[0.     0.3333 0.6667 1.    ]]
        >>> print(numpy.around(weights, 4))
        [0.375 1.125 1.125 0.375]
    """
    order = numpy.asarray(order, dtype=int).flatten()
    lower = numpy.asarray(lower).flatten()
    upper = numpy.asarray(upper).flatten()
    dim = max(lower.size, upper.size, order.size)
    order = numpy.ones(dim, dtype=int)*order
    lower = numpy.ones(dim)*lower
    upper = numpy.ones(dim)*upper

    results = [_newton_cotes(*args, growth=growth)
               for args in zip(order, lower, upper)]
    abscissas = [args[0] for args in results]
    abscissas = combine(abscissas).T
    weights = [args[1] for args in results]
    weights = numpy.prod(combine(weights), -1)
    return abscissas, weights


def _newton_cotes(order, lower, upper, growth):
    """Backend for Newton-Cotes quadrature rule."""
    if order == 0:
        return numpy.array([0.5*(lower+upper)]), numpy.ones(1)
    if growth:
        order = 2**order
    return (
        numpy.linspace(lower, upper, order+1),
        newton_cotes(order)[0]/(upper-lower),
    )
