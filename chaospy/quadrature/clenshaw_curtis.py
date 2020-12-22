"""Generate the quadrature nodes and weights in Clenshaw-Curtis quadrature."""
try:
    from functools import lru_cache
except ImportError:  # pragma: no coverage
    from functools32 import lru_cache

import numpy
import chaospy

from .hypercube import hypercube_quadrature


def quad_clenshaw_curtis(order, domain=(0., 1.), growth=False, segments=1):
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
        order (int, :class:`numpy.ndarray`):
            Quadrature order.
        domain (:class:`chaospy.Distribution`, :class:`numpy.ndarray`):
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
        abscissas (:class:`numpy.ndarray`):
            The quadrature points for where to evaluate the model function
            with ``abscissas.shape == (len(dist), steps)`` where ``steps`` is
            the number of samples.
        weights (:class:`numpy.ndarray`):
            The quadrature weights with ``weights.shape == (steps,)``.

    Notes:
        Implemented as proposed by Waldvogel :cite:`waldvogel_fast_2006`.

    Example:
        >>> abscissas, weights = quad_clenshaw_curtis(4, (0, 1))
        >>> abscissas.round(4)
        array([[0.    , 0.1464, 0.5   , 0.8536, 1.    ]])
        >>> weights.round(4)
        array([0.0333, 0.2667, 0.4   , 0.2667, 0.0333])
        >>> abscissas, weights = quad_clenshaw_curtis(4, (0, 1), segments=2)
        >>> abscissas.round(4)
        array([[0.  , 0.25, 0.5 , 0.75, 1.  ]])
        >>> weights.round(4)
        array([0.0833, 0.3333, 0.1667, 0.3333, 0.0833])

    """
    order = numpy.asarray(order)
    order = numpy.where(growth, numpy.where(order > 0, 2**order, 0), order)
    return hypercube_quadrature(
        quad_func=_clenshaw_curtis,
        order=order,
        domain=domain,
        segments=segments,
    )



@lru_cache(None)
def _clenshaw_curtis(order):
    """Backend for Clenshaw-Curtis quadrature."""
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
    assert numpy.isclose(numpy.sum(weights), 1)

    return abscissas, weights
