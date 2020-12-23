# -*- coding: utf-8 -*-
"""Generate the quadrature abscissas and weights in Fejér type I quadrature."""
try:
    from functools import lru_cache
except ImportError:  # pragma: no cover
    from functools32 import lru_cache
import numpy

from .hypercube import hypercube_quadrature


def quad_fejer_1(order, domain=(0, 1), growth=False, segments=1):
    """
    Generate the quadrature abscissas and weights in Fejér type I quadrature.

    Fejér proposed two quadrature rules very similar to
    :func:`chaospy.quad_clenshaw_curtis`. The only difference is that the
    endpoints are removed. That is, Fejér only used the interior extrema of the
    Chebyshev polynomials, i.e. the true stationary points. This makes this a
    better method for performing quadrature on infinite intervals, as the
    evaluation does not contain endpoint values.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (chaospy.Distribution, numpy.ndarray):
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
        abscissas (numpy.ndarray):
            The quadrature points for where to evaluate the model function
            with ``abscissas.shape == (len(dist), N)`` where ``N`` is the
            number of samples.
        weights (numpy.ndarray):
            The quadrature weights with ``weights.shape == (N,)``.

    Notes:
        Implemented as proposed by Waldvogel :cite:`waldvogel_fast_2006`.

    Example:
        >>> abscissas, weights = quad_fejer_1(3, (0, 1))
        >>> abscissas.round(4)
        array([[0.0381, 0.3087, 0.6913, 0.9619]])
        >>> weights.round(4)
        array([0.1321, 0.3679, 0.3679, 0.1321])
        >>> abscissas, weights = quad_fejer_1(3, (0, 1), segments=2)
        >>> abscissas.round(4)
        array([[0.0335, 0.25  , 0.4665, 0.5732, 0.9268]])
        >>> weights.round(4)
        array([0.1111, 0.2778, 0.1111, 0.25  , 0.25  ])

    """
    order = numpy.asarray(order)
    order = numpy.where(growth, 2*3**order-1, order)
    return hypercube_quadrature(
        quad_func=_fejer_type_1,
        order=order,
        domain=domain,
        segments=segments,
    )


@lru_cache(None)
def _fejer_type_1(order):
    """Backend for Fejer type I quadrature."""
    order = int(order)
    if order == 0:
        return numpy.array([.5]), numpy.array([1.])
    order += 1

    abscissas = -0.5*numpy.cos(numpy.pi*(numpy.arange(order)+0.5)/order)+0.5

    steps = numpy.arange(1, order, 2)
    length = len(steps)
    remains = order-length

    kappa = numpy.arange(remains)
    beta = numpy.hstack([2*numpy.exp(1j*numpy.pi*kappa/order)/(1-4*kappa**2),
                         numpy.zeros(length+1)])
    beta = beta[:-1]+numpy.conjugate(beta[:0:-1])

    weights = numpy.fft.ifft(beta)
    assert max(weights.imag) < 1e-15
    weights = weights.real/2.

    return abscissas, weights
