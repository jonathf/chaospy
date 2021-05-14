"""Generate quadrature abscissas and weights for discrete distributions."""
import numpy
import chaospy

from .hypercube import hypercube_quadrature


def discrete(order, domain=(0, 1), growth=False):
    """
    Generate quadrature abscissas and weights for discrete distributions.

    A specialized quadrature designed for discrete distributions. It is defined
    as an evenly spaced grid on the domain, rounded to nearest integer. Rule
    will converge to where all integer values on the domain is covered. This
    ensure that only necessary samples are evaluated.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (:class:`chaospy.Distribution`, numpy.ndarray):
            Either distribution or bounding of interval to integrate over.
        growth (bool):
            if true sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            The quadrature points and weights. The points are
            equi-spaced grid on the interior of the domain bounds.
            The weights are all equal to `1/len(weights[0])`.
            Either distribution or bounding of interval to integrate over.

    Examples:
        >>> distribution = chaospy.Binomial(6, 0.4)
        >>> abscissas, weights = chaospy.quadrature.discrete(4, distribution)
        >>> abscissas
        array([[0., 2., 3., 4., 6.]])
        >>> weights.round(4)
        array([0.0601, 0.4006, 0.3561, 0.178 , 0.0053])
        >>> abscissas, weights = chaospy.quadrature.discrete(10, distribution)
        >>> abscissas
        array([[0., 1., 2., 3., 4., 5., 6.]])
        >>> weights.round(4)
        array([0.0467, 0.1866, 0.311 , 0.2765, 0.1382, 0.0369, 0.0041])

    """
    order = numpy.asarray(order)
    order = numpy.where(growth, numpy.where(order > 0, 2**order, 0), order)
    return hypercube_quadrature(
        quad_func=discrete_simple,
        order=order,
        domain=domain,
        auto_scale=False,
    )


def discrete_simple(order, lower=-2, upper=2):
    """
    Backend for discrete quadrature.

    Use :func:`chaospy.quadrature.discrete` instead.
    """
    order = int(min(order, round(upper-lower)-1))
    abscissas = numpy.linspace(lower, upper, 2*order+3)[1::2].round()
    weights = numpy.full(order+1, (upper-lower)/(order+1.))
    return abscissas, weights
