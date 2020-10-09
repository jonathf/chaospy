"""Construct recurrence coefficients."""
import numpy

from .chebyshev import modified_chebyshev
from .jacobi import coefficients_to_quadrature
from .lanczos import lanczos
from .stieltjes import discretized_stieltjes

RECURRENCE_ALGORITHMS = ("analytical", "chebyshev", "lanczos", "stieltjes")


def construct_recurrence_coefficients(
        order,
        dist,
        rule="fejer",
        accuracy=200,
        recurrence_algorithm="",
):
    """
    Frontend wrapper for constructing *three terms recurrence* coefficients.

    The algorithm for constructing recurrence coefficients can be specified
    using the ``recurrence_algorithm`` flag. It accepts the strings:

    ``analytical``
        Some distributions have a built-in method for generating three terms
        recurrence coefficients. If that is not the case, raise an appropriate
        error. The most stable method, if available.
    ``chebyshev``
        Use modified Chebyshev algorithm to convert raw statistical moments to
        the recurrence coefficients. A good algorithm for when raw statistical
        moments are known analytically.
    ``lanczos``
        Use a known relationship between the Jakobi matrix and a matrix
        consisting of abscissas and weights from an alternative integration
        scheme to estimate the recurrence coefficients. Stabilized using ideas
        by Rutishauser. An alternative method to ``stieltjes``.
    ``stieltjes``
        Use the discretized Stieltjes algorithm for iterative estimate each
        recurrence coefficient using numerical integration. Typically the
        method known to have the highest convergence rate when no analytical
        information is available.

    Args:
        order (int):
            The order of the quadrature.
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution which density will be used as weight function.
            Assumed to one-dimensional.
        rule (str):
            In the case of ``lanczos`` or ``stieltjes``, defines the
            proxy-integration scheme.
        accuracy (int):
            In the case ``rule`` is used, defines the quadrature order of the
            scheme used. In practice, must be at least as large as ``order``.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights. If
            omitted, ``analytical`` will be tried first, and ``stieltjes`` used
            if that fails.

    Returns:
        (typing.List[numpy.ndarray]):
            List of recurrence coefficients with shape ``(2, order+1)``. The
            alpha and beta coefficients can found in ``out[0]`` and ``out[1]``
            respectively.

    Examples:
        >>> distribution = chaospy.Normal(0, 1)
        >>> coefficients = chaospy.construct_recurrence_coefficients(
        ...     4, distribution, recurrence_algorithm="stieltjes")
        >>> coefficients[0].round(3)
        array([[-0.,  0., -0.,  0., -0.],
               [ 1.,  1.,  2.,  3.,  4.]])
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
    import chaospy
    if not recurrence_algorithm:
        try:
            return construct_recurrence_coefficients(
                order, dist, rule, accuracy, recurrence_algorithm="analytical")
        except NotImplementedError:
            return construct_recurrence_coefficients(
                order, dist, rule, accuracy, recurrence_algorithm="stieltjes")

    if len(dist) > 1:
        orders = (order*numpy.ones(len(dist), dtype=int)).tolist()
        return [construct_recurrence_coefficients(
            int(order_), dist_, rule, accuracy, recurrence_algorithm)[0]
                for dist_, order_ in zip(dist, orders)]

    assert recurrence_algorithm in RECURRENCE_ALGORITHMS, (
        "recurrence algorithm '%s' not recognized" % recurrence_algorithm)
    assert not rule.startswith("gauss"), (
        "recursive Gaussian quadrature construct")

    if recurrence_algorithm == "analytical":
        coeffs = dist.ttr(numpy.arange(order+1, dtype=int))

    elif recurrence_algorithm == "chebyshev":
        moments = dist.mom(numpy.arange(2*(order+1), dtype=int))
        coeffs = modified_chebyshev(moments)

    elif recurrence_algorithm == "lanczos":
        from ..frontend import generate_quadrature
        abscissas, weights = generate_quadrature(
            accuracy, dist, rule=rule, segments=0)
        coeffs = lanczos(order, abscissas, weights)

    elif recurrence_algorithm == "stieltjes":
        from ..frontend import generate_quadrature
        abscissas, weights = generate_quadrature(
            accuracy, dist, rule=rule, segments=0)
        coeffs, _, _ = discretized_stieltjes(order, abscissas, weights)

    return [coeffs.reshape(2, int(order)+1)]
