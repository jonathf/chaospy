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
    ...     print(numpy.around(X, 2), numpy.around(W, 2))
    [[-1.]] [1.]
    [[-1.  1.]] [0.5 0.5]
    [[-1.   -0.38  0.38  1.  ]] [0.03 0.47 0.47 0.03]
    [[-1.   -0.69 -0.25  0.25  0.69  1.  ]]
     [0.01 0.15 0.35 0.35 0.15 0.01]

Multivariate samples::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Beta(4, 5))
    >>> X, W = chaospy.generate_quadrature(
    ...     2, distribution, rule="gauss_lobatto")
    >>> print(numpy.around(X, 3))  # doctest: +NORMALIZE_WHITESPACE
    [[-0.    -0.    -0.    -0.     0.276  0.276  0.276  0.276
       0.724  0.724  0.724  0.724  1.     1.     1.     1.   ]
     [ 0.     0.318  0.605  1.     0.     0.318  0.605  1.
       0.     0.318  0.605  1.     0.     0.318  0.605  1.   ]]
    >>> print(numpy.around(W, 3))  # doctest: +NORMALIZE_WHITESPACE
    [0.001 0.045 0.037 0.    0.006 0.224 0.184 0.002
     0.006 0.224 0.184 0.002 0.001 0.045 0.037 0.   ]
"""
from __future__ import print_function

import numpy
from scipy.linalg import solve_banded, solve

from .recurrence import (
    construct_recurrence_coefficients, coefficients_to_quadrature)
from .combine import combine


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
        dist (chaospy.distributions.baseclass.Dist):
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

    Raises:
        ValueError:
            Error raised if Loboto algorithm results in negative recurrence
            coefficients.

    Example:
        >>> abscissas, weights = quad_gauss_lobatto(
        ...     4, chaospy.Uniform(-1, 1))
        >>> print(numpy.around(abscissas, 3))
        [[-1.    -0.872 -0.592 -0.209  0.209  0.592  0.872  1.   ]]
        >>> print(numpy.around(weights, 3))
        [0.018 0.105 0.171 0.206 0.206 0.171 0.105 0.018]
    """
    assert not rule.startswith("gauss"), "recursive Gaussian quadrature call"
    lower, upper = dist.range()
    if order == 0:
        return lower.reshape(1, -1), numpy.array([1.])

    coefficients = construct_recurrence_coefficients(
        2*order-1, dist, rule, accuracy, recurrence_algorithm)
    coefficients = [_lobatto(coeffs, lo, up)
                    for coeffs, lo, up in zip(coefficients, lower, upper)]
    abscissas, weights = coefficients_to_quadrature(coefficients)

    abscissas = combine(abscissas).T.reshape(len(dist), -1)
    weights = numpy.prod(combine(weights), -1)

    return abscissas, weights


def _lobatto(coeffs, xl1, xl2):
    """
    Compute the Lobatto nodes and weights with the preassigned node xl1, xl2.
    Based on the section 7 of the paper

        Some modified matrix eigenvalue problems,
        Gene Golub,
        SIAM Review Vol 15, No. 2, April 1973, pp.318--334,

    and

        http://www.scientificpython.net/pyblog/radau-quadrature
    """
    alpha = numpy.array(coeffs[0])
    beta = numpy.array(coeffs[1])
    en = numpy.zeros(len(alpha)-1)
    en[-1] = 1
    A1 = numpy.vstack((numpy.sqrt(beta), alpha - xl1))
    J1 = numpy.vstack((A1[:, 0:-1], A1[0, 1:]))
    A2 = numpy.vstack((numpy.sqrt(beta), alpha - xl2))
    J2 = numpy.vstack((A2[:, 0:-1], A2[0, 1:]))
    g1 = solve_banded((1, 1), J1, en)
    g2 = solve_banded((1, 1), J2, en)
    C = numpy.array(((1, -g1[-1]), (1, -g2[-1])))
    xl = numpy.array((xl1, xl2))
    ab = solve(C, xl)

    alpha[-1] = ab[0]
    beta[-1] = ab[1]
    return numpy.array([alpha, beta])
