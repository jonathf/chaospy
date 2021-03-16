"""Gauss-Gegenbauer quadrature rule."""
import numpy
import chaospy

from .hypercube import hypercube_quadrature


def gegenbauer(order, alpha, lower=-1, upper=1, physicist=False):
    """
    Gauss-Gegenbauer quadrature rule.

    Compute the sample points and weights for Gauss-Gegenbauer quadrature. The
    sample points are the roots of the nth degree Gegenbauer polynomial. These
    sample points and weights correctly integrate polynomials of degree
    :math:`2N-1` or less.

    Gaussian quadrature come in two variants: physicist and probabilist. For
    Gauss-Gegenbauer physicist means a weight function
    :math:`(1-x^2)^{\alpha-0.5}` and weights that sum to :math`2^{2\alpha-1}`,
    and probabilist means a weight function is
    :math:`B(\alpha+0.5, \alpha+0.5) (x-x^2)^{\alpha+1/2}` (where :math:`B` is
    the beta normalizing constant) which sum to 1.

    Args:
        order (int):
            The quadrature order.
        alpha (float):
            Gegenbauer shape parameter.
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
        >>> abscissas, weights = chaospy.quadrature.gegenbauer(3, alpha=2)
        >>> abscissas
        array([[-0.72741239, -0.26621648,  0.26621648,  0.72741239]])
        >>> weights
        array([0.10452141, 0.39547859, 0.39547859, 0.10452141])

    See also:
        :func:`chaospy.quadrature.gaussian`

    """
    order = int(order)
    coefficients = chaospy.construct_recurrence_coefficients(
        order=order, dist=chaospy.Beta(alpha+0.5, alpha+0.5, lower, upper))
    [abscissas], [weights] = chaospy.coefficients_to_quadrature(coefficients)
    weights *= 2**(2*alpha-1) if physicist else 1
    return abscissas[numpy.newaxis], weights
