"""Generate the quadrature nodes and weights in Clenshaw-Curtis quadrature."""
try:
    from functools import lru_cache
except ImportError:  # pragma: no coverage
    from functools32 import lru_cache

import numpy
import chaospy

from .hypercube import hypercube_quadrature


def clenshaw_curtis(order, domain=(0., 1.), growth=False, segments=1):
    """
    Generate the quadrature nodes and weights in Clenshaw-Curtis quadrature.

    Clenshaw-Curtis quadrature method is a good all-around quadrature method
    comparable to Gaussian quadrature, but typically limited to finite
    intervals without a specific weight function. In addition to be quite
    accurate, the weights and abscissas can be calculated quite fast.

    Another thing to note is that Clenshaw-Curtis, with an appropriate growth
    rule is fully nested. This means, if one applies a method that combines
    different order of quadrature rules, the number of evaluations can often be
    reduced as the abscissas can be used across levels.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (:class:`chaospy.Distribution`, numpy.ndarray):
            Either distribution or bounding of interval to integrate over.
        growth (bool):
            If True sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples.
        segments (int):
            Split intervals into steps subintervals and create a patched
            quadrature based on the segmented quadrature. Can not be lower than
            `order`. If 0 is provided, default to square root of `order`.
            Nested samples only appear when the number of segments are fixed.

    Returns:
        abscissas (numpy.ndarray):
            The quadrature points for where to evaluate the model function
            with ``abscissas.shape == (len(dist), steps)`` where ``steps`` is
            the number of samples.
        weights (numpy.ndarray):
            The quadrature weights with ``weights.shape == (steps,)``.

    Notes:
        Implemented as proposed by Waldvogel :cite:`waldvogel_fast_2006`.

    Example:
        >>> abscissas, weights = chaospy.quadrature.clenshaw_curtis(4, (0, 1))
        >>> abscissas.round(4)
        array([[0.    , 0.1464, 0.5   , 0.8536, 1.    ]])
        >>> weights.round(4)
        array([0.0333, 0.2667, 0.4   , 0.2667, 0.0333])

    See also:
        :func:`chaospy.quadrature.gaussian`
        :func:`chaospy.quadrature.fejer_1`
        :func:`chaospy.quadrature.fejer_2`

    """
    order = numpy.asarray(order)
    order = numpy.where(growth, numpy.where(order > 0, 2**order, 0), order)
    return hypercube_quadrature(
        quad_func=clenshaw_curtis_simple,
        order=order,
        domain=domain,
        segments=segments,
    )


@lru_cache(None)
def clenshaw_curtis_simple(order):
    """
    Backend for Clenshaw-Curtis quadrature.

    Use :func:`chaospy.quadrature.clenshaw_curtis` instead.
    """
    order = int(order)
    if order == 0:
        return numpy.array([.5]), numpy.array([1.])
    elif order == 1:
        return numpy.array([0., 1.]), numpy.array([0.5, 0.5])

    theta = (order-numpy.arange(order+1))*numpy.pi/order
    abscissas = 0.5*numpy.cos(theta)+0.5

    steps = numpy.arange(1, order, 2)
    length = len(steps)
    remains = order-length

    beta = numpy.hstack([2./(steps*(steps-2)), [1./steps[-1]], numpy.zeros(remains)])
    beta = -beta[:-1]-beta[:0:-1]

    gamma = -numpy.ones(order)
    gamma[length] += order
    gamma[remains] += order
    gamma /= (order**2-1+(order%2))

    weights = numpy.fft.ihfft(beta+gamma)
    assert max(weights.imag) < 1e-15
    weights = weights.real
    weights = numpy.hstack([weights, weights[len(weights)-2+(order%2)::-1]])/2

    return abscissas, weights
