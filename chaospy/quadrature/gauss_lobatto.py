"""
Gauss-Radau formula for numerical estimation of integrals. It requires
:math:`m+1` points and fits all Polynomials to degree :math:`2m`, so it
effectively fits exactly all Polynomials of degree :math:`2m-3`.

Gauss-Radau is defined by having two abscissas to be fixed to the endpoints,
while the others are built around these points. So if a distribution is defined
on the interval ``(a, b)``, then both ``a`` and ``b`` are abscissas in this
scheme. Note though that this does not always possible to achieve in practice,
and an error might be raised.

Example usage
-------------

With increasing order::

    >>> distribution = chaospy.Beta(2, 2, lower=-1, upper=1)
    >>> for order in range(4):  # doctest: +NORMALIZE_WHITESPACE
    ...     X, W = chaospy.generate_quadrature(
    ...         order, distribution, rule="gauss_lobatto")
    ...     print(X.round(2), W.round(2))
    [[-1.]] [1.]
    [[-1.  1.]] [0.5 0.5]
    [[-1.   -0.38  0.38  1.  ]] [0.03 0.47 0.47 0.03]
    [[-1.   -0.69 -0.25  0.25  0.69  1.  ]]
     [0.01 0.15 0.35 0.35 0.15 0.01]

Multivariate samples::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Beta(4, 5))
    >>> X, W = chaospy.generate_quadrature(
    ...     2, distribution, rule="gauss_lobatto")
    >>> X.round(3)
    array([[-0.   , -0.   , -0.   , -0.   ,  0.276,  0.276,  0.276,  0.276,
             0.724,  0.724,  0.724,  0.724,  1.   ,  1.   ,  1.   ,  1.   ],
           [ 0.   ,  0.318,  0.605,  1.   ,  0.   ,  0.318,  0.605,  1.   ,
             0.   ,  0.318,  0.605,  1.   ,  0.   ,  0.318,  0.605,  1.   ]])
    >>> W.round(3)
    array([0.001, 0.045, 0.037, 0.   , 0.006, 0.224, 0.184, 0.002, 0.006,
           0.224, 0.184, 0.002, 0.001, 0.045, 0.037, 0.   ])
"""
import numpy
from scipy.linalg import solve_banded, solve

from .recurrence import (
    construct_recurrence_coefficients, coefficients_to_quadrature)
from .combine import combine_quadrature


def quad_gauss_lobatto(
        order,
        dist,
        rule="fejer",
        accuracy=100,
        recurrence_algorithm="",
):
    """
    Generate the abscissas and weights in Gauss-Loboto quadrature.

    Args:
        order (int):
            Quadrature order.
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution weights to be used to create higher order nodes
            from.
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
        (numpy.ndarray, numpy.ndarray):
            abscissas:
                The quadrature points for where to evaluate the model function
                with ``abscissas.shape == (len(dist), N)`` where ``N`` is the
                number of samples.
            weights:
                The quadrature weights with ``weights.shape == (N,)``.

    Example:
        >>> abscissas, weights = quad_gauss_lobatto(
        ...     4, chaospy.Uniform(-1, 1))
        >>> abscissas.round(3)
        array([[-1.   , -0.872, -0.592, -0.209,  0.209,  0.592,  0.872,  1.   ]])
        >>> weights.round(3)
        array([0.018, 0.105, 0.171, 0.206, 0.206, 0.171, 0.105, 0.018])
    """
    assert not rule.startswith("gauss"), "recursive Gaussian quadrature call"
    if order == 0:
        return dist.lower.reshape(1, -1), numpy.array([1.])

    coefficients = construct_recurrence_coefficients(
        2*order-1, dist, rule, accuracy, recurrence_algorithm)
    coefficients = [_lobatto(coeffs, (lo, up))
                    for coeffs, lo, up in zip(coefficients, dist.lower, dist.upper)]
    abscissas, weights = coefficients_to_quadrature(coefficients)

    return combine_quadrature(abscissas, weights)


def _lobatto(coefficients, preassigned):
    """
    Compute the Lobatto nodes and weights with the preassigned value pair.
    Based on the section 7 of the paper

        Some modified matrix eigenvalue problems,
        Gene Golub,
        SIAM Review Vol 15, No. 2, April 1973, pp.318--334,

    and

        http://www.scientificpython.net/pyblog/radau-quadrature

    Args:
        coefficients (numpy.ndarray):
            Three terms recurrence coefficients.
        preassigned (Tuple[float, float]):
            Values that are assume to be fixed.
    """
    alpha = numpy.array(coefficients[0])
    beta = numpy.array(coefficients[1])
    vec_en = numpy.zeros(len(alpha)-1)
    vec_en[-1] = 1
    mat_a1 = numpy.vstack((numpy.sqrt(beta), alpha-preassigned[0]))
    mat_j1 = numpy.vstack((mat_a1[:, 0:-1], mat_a1[0, 1:]))
    mat_a2 = numpy.vstack((numpy.sqrt(beta), alpha - preassigned[1]))
    mat_j2 = numpy.vstack((mat_a2[:, 0:-1], mat_a2[0, 1:]))
    mat_g1 = solve_banded((1, 1), mat_j1, vec_en)
    mat_g2 = solve_banded((1, 1), mat_j2, vec_en)
    mat_c = numpy.array(((1, -mat_g1[-1]), (1, -mat_g2[-1])))
    alpha[-1], beta[-1] = solve(mat_c, preassigned)

    return numpy.array([alpha, beta])
