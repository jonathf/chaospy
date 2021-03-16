"""Generalized Gauss-Laguerre quadrature rule."""
import numpy
from scipy.special import gamma
import chaospy

from .hypercube import hypercube_quadrature


def laguerre(order, alpha=0., physicist=False):
    r"""
    Generalized Gauss-Laguerre quadrature rule.

    Compute the sample points and weights for Gauss-Laguerre quadrature. The
    sample points are the roots of the nth degree Laguerre polynomial. These
    sample points and weights correctly integrate polynomials of degree
    :math:`2N-1` or less.

    Gaussian quadrature come in two variants: physicist and probabilist. For
    Gauss-Laguerre physicist means a weight function :math:`x^\alpha e^{-x}`
    and weights that sum to :math`\Gamma(\alpha+1)`, and probabilist means a
    weight function is :math:`x^\alpha e^{-x}` and sum to 1.

    Args:
        order (int):
            The quadrature order.
        alpha (float):
            Shape parameter. Defaults to non-generalized Laguerre if 0.
        physicist (bool):
            Use physicist weights instead of probabilist.

    Returns:
        abscissas (numpy.ndarray):
            The ``order+1`` quadrature points for where to evaluate the model
            function with.
        weights (numpy.ndarray):
            The quadrature weights associated with each abscissas.

    Examples:
        >>> abscissas, weights = chaospy.quadrature.laguerre(2)
        >>> abscissas
        array([[0.41577456, 2.29428036, 6.28994508]])
        >>> weights
        array([0.71109301, 0.27851773, 0.01038926])

    See also:
        :func:`chaospy.quadrature.gaussian`

    """
    order = int(order)
    coefficients = chaospy.construct_recurrence_coefficients(
        order=order, dist=chaospy.Gamma(alpha+1))
    [abscissas], [weights] = chaospy.coefficients_to_quadrature(coefficients)
    weights *= gamma(alpha+1) if physicist else 1
    return abscissas[numpy.newaxis], weights
