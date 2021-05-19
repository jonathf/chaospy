"""Construct recurrence coefficients."""
import numpy
import chaospy

from .chebyshev import modified_chebyshev
from .jacobi import coefficients_to_quadrature
from .lanczos import lanczos
from .stieltjes import stieltjes

RECURRENCE_ALGORITHMS = ("chebyshev", "lanczos", "stieltjes")


def construct_recurrence_coefficients(
        order,
        dist,
        recurrence_algorithm="stieltjes",
        rule="clenshaw_curtis",
        tolerance=1e-10,
        scaling=3,
        n_max=5000,
):
    """
    Frontend wrapper for constructing *three terms recurrence* coefficients.

    The algorithm for constructing recurrence coefficients can be specified
    using the ``recurrence_algorithm`` flag. It accepts the strings:

    ``stieltjes``
        Use the discretized Stieltjes algorithm for iterative estimate each
        recurrence coefficient using numerical integration. Typically the
        method known to have the highest convergence rate.
    ``chebyshev``
        Use modified Chebyshev algorithm to convert raw statistical moments to
        the recurrence coefficients. A good algorithm for when raw statistical
        moments are known analytically.
    ``lanczos``
        Use a known relationship between the Jakobi matrix and a matrix
        consisting of abscissas and weights from an alternative integration
        scheme to estimate the recurrence coefficients. Stabilized using ideas
        by Rutishauser. An alternative method to ``stieltjes``.

    Args:
        order (int):
            The order of the quadrature.
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution which density will be used as weight function.
            Assumed to one-dimensional.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights.
        rule (str):
            In the case of ``lanczos`` or ``stieltjes``, defines the
            proxy-integration scheme.
        tolerance (float):
            The allowed relative error in norm between two quadrature orders
            before method assumes convergence.
        scaling (float):
            A multiplier the adaptive order increases with for each step
            quadrature order is not converged. Use 0 to indicate unit
            increments.
        n_max (int):
            The allowed number of quadrature points to use in approximation.

    Returns:
        (typing.List[numpy.ndarray]):
            List of recurrence coefficients with shape ``(2, order+1)``. The
            alpha and beta coefficients can found in ``out[0]`` and ``out[1]``
            respectively.

    Examples:
        >>> distribution = chaospy.Normal(1, 1)
        >>> coefficients = chaospy.construct_recurrence_coefficients(
        ...     4, distribution, recurrence_algorithm="stieltjes")
        >>> coefficients[0].round(3)
        array([[1., 1., 1., 1., 1.],
               [1., 1., 2., 3., 4.]])
        >>> distribution = chaospy.J(chaospy.Exponential(), chaospy.Uniform())
        >>> coefficients = chaospy.construct_recurrence_coefficients(
        ...     [2, 4], distribution, recurrence_algorithm="chebyshev")
        >>> coefficients[0].round(4)
        array([[1., 3., 5.],
               [1., 1., 4.]])
        >>> coefficients[1].round(4)
        array([[0.5   , 0.5   , 0.5   , 0.5   , 0.5   ],
               [1.    , 0.0833, 0.0667, 0.0643, 0.0635]])
    """
    assert isinstance(dist, chaospy.Distribution), (
        "%s is not a distribution" % str(dist))
    if len(dist) > 1:
        orders = (order*numpy.ones(len(dist), dtype=int)).tolist()
        return [construct_recurrence_coefficients(
            order=int(order_),
            dist=dist_,
            recurrence_algorithm=recurrence_algorithm,
            rule=rule,
            tolerance=tolerance,
            scaling=scaling,
            n_max=n_max,
        )[0] for dist_, order_ in zip(dist, orders)]

    assert recurrence_algorithm in RECURRENCE_ALGORITHMS, (
        "recurrence algorithm '%s' not recognized" % recurrence_algorithm)
    assert not rule.startswith("gauss"), (
        "recursive Gaussian quadrature construct")

    if recurrence_algorithm == "chebyshev":
        moments = dist.mom(numpy.arange(2*(order+1), dtype=int))
        coeffs = modified_chebyshev(moments)

    elif recurrence_algorithm == "lanczos":
        coeffs = lanczos(order, dist, rule=rule, tolerance=tolerance)

    elif recurrence_algorithm == "stieltjes":
        coeffs, _, _ = stieltjes(order, dist, rule=rule, tolerance=tolerance)

    return [coeffs.reshape(2, int(order)+1)]
