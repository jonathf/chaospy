# -*- coding: utf-8 -*-
"""Generate the quadrature abscissas and weights in Fejer quadrature."""
import numpy
import chaospy

from .hypercube import hypercube_quadrature
from .clenshaw_curtis import clenshaw_curtis_simple


def fejer_2(order, domain=(0, 1), growth=False, segments=1):
    """
    Generate the quadrature abscissas and weights in Fejér type II quadrature.

    Fejér proposed two quadrature rules very similar to
    :func:`chaospy.quadrature.clenshaw_curtis`. The only difference is that the
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
        >>> abscissas, weights = chaospy.quadrature.fejer_2(3, (0, 1))
        >>> abscissas.round(4)
        array([[0.0955, 0.3455, 0.6545, 0.9045]])
        >>> weights.round(4)
        array([0.1804, 0.2996, 0.2996, 0.1804])

    See also:
        :func:`chaospy.quadrature.gaussian`
        :func:`chaospy.quadrature.clenshaw_curtis`
        :func:`chaospy.quadrature.fejer_1`

    """
    order = numpy.asarray(order)
    order = numpy.where(growth, numpy.where(order > 0, 2**(order+1)-2, 0), order)
    return hypercube_quadrature(
        quad_func=fejer_2_simple,
        order=order,
        domain=domain,
        segments=segments,
    )


def fejer_2_simple(order):
    """
    Backend for Fejer type II quadrature.

    Same as Clenshaw-Curtis, but with the end nodes removed.
    """
    abscissas, weights = clenshaw_curtis_simple(order+2)
    return abscissas[1:-1], weights[1:-1]
