"""Gauss-Hermite quadrature rule."""
import numpy
import chaospy

from .hypercube import hypercube_quadrature


def hermite(order, mu=0., sigma=1., physicist=False):
    r"""
    Gauss-Hermite quadrature rule.

    Compute the sample points and weights for Gauss-Hermite quadrature. The
    sample points are the roots of the nth degree Hermite polynomial. These
    sample points and weights correctly integrate polynomials of degree
    :math:`2N-1` or less.

    Gaussian quadrature come in two variants: physicist and probabilist. For
    Gauss-Hermite physicist means a weight function :math:`e^{-x^2}` and
    weights that sum to :math`\sqrt(\pi)`, and probabilist means a weight
    function is :math:`e^{-x^2/2}` and sum to 1.

    Args:
        order (int):
            The quadrature order.
        mu (float):
            Non-centrality parameter.
        sigma (float):
            Scale parameter.
        physicist (bool):
            Use physicist weights instead of probabilist variant.

    Returns:
        abscissas (numpy.ndarray):
            The ``order+1`` quadrature points for where to evaluate the model
            function with.
        weights (numpy.ndarray):
            The quadrature weights associated with each abscissas.

    Examples:
        >>> abscissas, weights = chaospy.quadrature.hermite(3)
        >>> abscissas
        array([[-2.33441422, -0.74196378,  0.74196378,  2.33441422]])
        >>> weights
        array([0.04587585, 0.45412415, 0.45412415, 0.04587585])

    See also:
        :func:`chaospy.quadrature.gaussian`

    """
    order = int(order)
    sigma = float(sigma*2**-0.5 if physicist else sigma)
    coefficients = chaospy.construct_recurrence_coefficients(
        order=order, dist=chaospy.Normal(0, sigma))
    [abscissas], [weights] = chaospy.coefficients_to_quadrature(coefficients)
    weights = weights*numpy.pi**0.5 if physicist else weights
    if order%2 == 0:
        abscissas[len(abscissas)//2] = 0
    abscissas += mu
    return abscissas[numpy.newaxis], weights
