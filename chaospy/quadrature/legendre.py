"""Gauss-Legendre quadrature rule."""
try:
    from functools import lru_cache
except ImportError:  # pragma: no cover
    from functools32 import lru_cache

import numpy
import chaospy

from .hypercube import hypercube_quadrature


def legendre(order, lower=-1., upper=1., physicist=False):
    """
    Gauss-Legendre quadrature rule.

    Compute the sample points and weights for Gauss-Legendre quadrature. The
    sample points are the roots of the N-th degree Legendre polynomial. These
    sample points and weights correctly integrate polynomials of degree
    :math:`2N-1` or less over the interval ``[lower, upper]``.

    Gaussian quadrature come in two variants: physicist and probabilist. For
    Gauss-Legendre physicist means a weight function constant 1 and weights sum
    to ``upper-lower``, and probabilist means weight function constant
    ``1/(upper-lower)`` while weights sum to 1.

    Args:
        order (int):
            The quadrature order.
        lower (float):
            Lower bound for the integration interval.
        upper (float):
            Upper bound for the integration interval.
        physicist (bool):
            Use physicist weights instead of probabilist.

    Returns:
        abscissas (numpy.ndarray):
            The ``order+1`` quadrature points for where to evaluate the model
            function with.
        weights (numpy.ndarray):
            The quadrature weights associated with each abscissas.

    Examples:
        >>> abscissas, weights = chaospy.quadrature.legendre(2)
        >>> abscissas
        array([[-0.77459667,  0.        ,  0.77459667]])
        >>> weights
        array([0.27777778, 0.44444444, 0.27777778])

    See also:
        :func:`chaospy.quadrature.gaussian`
        :func:`chaospy.quadrature.legendre_proxy`

    """
    abscissas, weights = hypercube_quadrature(
        legendre_simple,
        order=int(order),
        domain=(float(lower), float(upper)),
    )
    weights = weights if physicist else weights/(upper-lower)
    return abscissas, weights


def legendre_proxy(
        order,
        domain=(0, 1),
        segments=1,
):
    r"""
    Generate proxy abscissas and weights from Legendre quadrature.

    Legendre provides optimal abscissas :math:`X_i` and weights :math:`W_i` to
    solve the integration problem:

    .. math::

        \int f(x) dx \approx \sum W_i f(X_i)

    over a function :math:`f`, where the probability density function :math:`p`
    is uniform.

    Since the weight function is constant, it can in principle be used to
    integrate any density function by considering it a part of the function. In
    other words:

    .. math::

        \int p(x) f(x) \approx \sum W_i p(X_i) f(X_i) = \sum W_i^' f(X_i)

    So when providing non-uniform distribution as `domain`, the weights will be
    adjusted with:

    .. math::

        W_i^' = W_i p(X_i)

    Bounds of the Legendre schemes is chosen to be the same as the distribution
    provided. This makes it a bad choice for unbound distributions.

    To get optimal abscissas and weights directly from a density, use
    :func:`chaospy.quadrature.gaussian` instead.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (:class:`chaospy.Distribution`, numpy.ndarray):
            Either distribution or bounding of interval to integrate over.
        segments (int):
            Split intervals into steps subintervals and create a patched
            quadrature based on the segmented quadrature. Can not be lower than
            `order`. If 0 is provided, default to square root of `order`.
            Nested samples only appear when the number of segments are fixed.

    Returns:
        abscissas (numpy.ndarray):
            The quadrature points for where to evaluate the model function with
            ``abscissas.shape == (len(dist), N)`` where ``N`` is the number of
            samples.
        weights (numpy.ndarray):
            The quadrature weights with ``weights.shape == (N,)``.

    Example:
        >>> abscissas, weights = chaospy.quadrature.legendre_proxy(3)
        >>> abscissas.round(4)
        array([[0.0694, 0.33  , 0.67  , 0.9306]])
        >>> weights.round(4)
        array([0.1739, 0.3261, 0.3261, 0.1739])
        >>> abscissas, weights = chaospy.quadrature.legendre_proxy(3, chaospy.Uniform(0, 1))
        >>> abscissas.round(4)
        array([[0.0694, 0.33  , 0.67  , 0.9306]])
        >>> weights.round(4)
        array([0.1739, 0.3261, 0.3261, 0.1739])
        >>> abscissas, weights = chaospy.quadrature.legendre_proxy(3, chaospy.Beta(2, 2))
        >>> abscissas.round(4)
        array([[0.0694, 0.33  , 0.67  , 0.9306]])
        >>> weights.round(4)
        array([0.0674, 0.4326, 0.4326, 0.0674])

    See also:
        :func:`chaospy.quadrature.legendre`,
        :func:`chaospy.quadrature.gaussian`

    """
    return hypercube_quadrature(
        legendre_simple,
        order=order,
        domain=domain,
        segments=segments,
    )


@lru_cache(None)
def legendre_simple(order):
    """
    Simple Legendre quadrature on the [0, 1] interval.

    Use :func:`chaospy.quadrature.legendre` instead.
    """
    coefficients = chaospy.construct_recurrence_coefficients(
        order=int(order), dist=chaospy.Uniform(-1, 1))
    [abscissas], [weights] = chaospy.coefficients_to_quadrature(coefficients)
    return abscissas*0.5+0.5, weights
