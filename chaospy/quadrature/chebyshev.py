"""Chebyshev-Gauss quadrature rule of the first kind."""
import numpy
import chaospy

from .hypercube import hypercube_quadrature


def chebyshev_1(order, lower=-1, upper=1, physicist=False):
    r"""
    Chebyshev-Gauss quadrature rule of the first kind.

    Compute the sample points and weights for Chebyshev-Gauss quadrature. The
    sample points are the roots of the nth degree Chebyshev polynomial. These
    sample points and weights correctly integrate polynomials of degree
    :math:`2N-1` or less.

    Gaussian quadrature come in two variants: physicist and probabilist. For
    first order Chebyshev-Gauss physicist means a weight function
    :math:`1/\sqrt{1-x^2}` and weights that sum to :math`1/2`, and probabilist
    means a weight function is :math:`1/\sqrt{x (1-x)}` and sum to 1.

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
        >>> abscissas, weights = chaospy.quadrature.chebyshev_1(3)
        >>> abscissas
        array([[-0.92387953, -0.38268343,  0.38268343,  0.92387953]])
        >>> weights
        array([0.25, 0.25, 0.25, 0.25])

    See also:
        :func:`chaospy.quadrature.chebyshev_2`
        :func:`chaospy.quadrature.gaussian`

    """
    order = int(order)
    coefficients = chaospy.construct_recurrence_coefficients(
        order=order, dist=chaospy.Beta(0.5, 0.5, lower, upper))
    [abscissas], [weights] = chaospy.coefficients_to_quadrature(coefficients)
    weights *= 0.5 if physicist else 1
    return abscissas[numpy.newaxis], weights



def chebyshev_2(order, lower=-1, upper=1, physicist=False):
    r"""
    Chebyshev-Gauss quadrature rule of the second kind.

    Compute the sample points and weights for Chebyshev-Gauss quadrature. The
    sample points are the roots of the nth degree Chebyshev polynomial. These
    sample points and weights correctly integrate polynomials of degree
    :math:`2N-1` or less.

    Gaussian quadrature come in two variants: physicist and probabilist. For
    second order Chebyshev-Gauss physicist means a weight function
    :math:`\sqrt{1-x^2}` and weights that sum to :math`2`, and probabilist
    means a weight function is :math:`\sqrt{x (1-x)}` and sum to 1.

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
        >>> abscissas, weights = chaospy.quadrature.chebyshev_2(3)
        >>> abscissas
        array([[-0.80901699, -0.30901699,  0.30901699,  0.80901699]])
        >>> weights
        array([0.1381966, 0.3618034, 0.3618034, 0.1381966])

    See also:
        :func:`chaospy.quadrature.chebyshev_1`
        :func:`chaospy.quadrature.gaussian`

    """
    order = int(order)
    coefficients = chaospy.construct_recurrence_coefficients(
        order=order, dist=chaospy.Beta(1.5, 1.5, lower, upper))
    [abscissas], [weights] = chaospy.coefficients_to_quadrature(coefficients)
    weights *= 2 if physicist else 1
    return abscissas[numpy.newaxis], weights
