"""Gauss-Jakobi quadrature rule."""
import numpy
import chaospy

from .hypercube import hypercube_quadrature


def jacobi(order, alpha, beta, lower=-1, upper=1, physicist=False):
    """
    Gauss-Jacobi quadrature rule.

    Compute the sample points and weights for Gauss-Jacobi quadrature. The
    sample points are the roots of the nth degree Jacobi polynomial. These
    sample points and weights correctly integrate polynomials of degree
    :math:`2N-1` or less.

    Gaussian quadrature come in two variants: physicist and probabilist. For
    Gauss-Jacobi physicist means a weight function
    :math:`(1-x)^\alpha (1+x)^\beta` and
    weights that sum to :math`2^{\alpha+\beta}`, and probabilist means a weight
    function is :math:`B(\alpha, \beta) x^{\alpha-1}(1-x)^{\beta-1}` (where
    :math:`B` is the beta normalizing constant) which sum to 1.

    Args:
        order (int):
            The quadrature order.
        alpha (float):
            First Jakobi shape parameter.
        beta (float):
            Second Jakobi shape parameter.
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
        >>> abscissas, weights = chaospy.quadrature.jacobi(3, alpha=2, beta=2)
        >>> abscissas
        array([[-0.69474659, -0.25056281,  0.25056281,  0.69474659]])
        >>> weights
        array([0.09535261, 0.40464739, 0.40464739, 0.09535261])

    See also:
        :func:`chaospy.quadrature.gaussian`

    """
    order = int(order)
    coefficients = chaospy.construct_recurrence_coefficients(
        order=order, dist=chaospy.Beta(alpha+1, beta+1, lower, upper))
    [abscissas], [weights] = chaospy.coefficients_to_quadrature(coefficients)
    weights *= 2**(alpha+beta) if physicist else 1
    return abscissas[numpy.newaxis], weights
