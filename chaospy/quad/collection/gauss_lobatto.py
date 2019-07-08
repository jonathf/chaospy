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
    ...     X, W = chaospy.generate_quadrature(order, distribution, rule="L")
    ...     print("{} {}".format(numpy.around(X, 2), numpy.around(W, 2)))
    [[-1.]] [1.]
    [[-1.  1.]] [0.5 0.5]
    [[-1.   -0.38  0.38  1.  ]] [0.03 0.47 0.47 0.03]
    [[-1.   -0.69 -0.25  0.25  0.69  1.  ]] [0.01 0.15 0.35 0.35 0.15 0.01]

Multivariate samples::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Beta(4, 5))
    >>> X, W = chaospy.generate_quadrature(2, distribution, rule="L")
    >>> print(numpy.around(X, 3))
    [[-0.    -0.    -0.    -0.     0.276  0.276  0.276  0.276  0.724  0.724
       0.724  0.724  1.     1.     1.     1.   ]
     [ 0.     0.318  0.605  1.     0.     0.318  0.605  1.     0.     0.318
       0.605  1.     0.     0.318  0.605  1.   ]]
    >>> print(numpy.around(W, 3))
    [0.001 0.045 0.037 0.    0.006 0.224 0.184 0.002 0.006 0.224 0.184 0.002
     0.001 0.045 0.037 0.   ]
"""
import numpy
from scipy.linalg import solve_banded, solve

from .golub_welsch import _golub_welsch
from ..stieltjes import generate_stieltjes
from ..combine import combine


def quad_gauss_lobatto(order, dist=None):
    """
    Generate the abscissas and weights in Gauss-Loboto quadrature.

    Args:
        order (int):
            Quadrature order.
        dist (chaospy.distributions.baseclass.Dist):
            The distribution weights to be used to create higher order nodes
            from. If omitted, use ``Uniform(-1, 1)``.

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
        >>> abscissas, weights = quad_gauss_lobatto(4)
        >>> print(numpy.around(abscissas, 3))
        [[-1.    -0.872 -0.592 -0.209  0.209  0.592  0.872  1.   ]]
        >>> print(numpy.around(weights, 3))
        [0.018 0.105 0.171 0.206 0.206 0.171 0.105 0.018]
    """
    if dist is None:
        from chaospy.distributions.collection import Uniform
        dist = Uniform(lower=-1, upper=1)

    lower, upper = dist.range()
    if order == 0:
        return lower.reshape(-1, 1), numpy.array([1.])

    _, _, coeffs_a, coeffs_b = generate_stieltjes(dist, 2*order-1, retall=True)

    results = numpy.array([_lobatto(*coeffs) for coeffs in zip(
        coeffs_a, coeffs_b, lower, upper)])
    coeffs_a, coeffs_b = results[:, 0], results[:, 1]

    if numpy.any(coeffs_b < 0):
        raise ValueError(
            "Lobatto algorithm results in illegal coefficients;\n"
            "Gauss-Lobatto possibly not possible for %s" % dist
        )

    # Solve eigen problem for a tridiagonal matrix with As and Bs
    abscissas, weights = _golub_welsch(
        [len(coeffs_a[0])]*len(dist), coeffs_a, coeffs_b)
    abscissas = combine(abscissas).T
    weights = numpy.prod(combine(weights), -1)
    return abscissas, weights


def _lobatto(alpha, beta, xl1, xl2):
    """Compute the Lobatto nodes and weights with the preassigned node xl1, xl2.
    Based on the section 7 of the paper

        Some modified matrix eigenvalue problems,
        Gene Golub,
        SIAM Review Vol 15, No. 2, April 1973, pp.318--334,

    and

        http://www.scientificpython.net/pyblog/radau-quadrature
    """
    assert alpha.shape == beta.shape
    if len(alpha.shape) == 2:
        coeffs = numpy.array([_lobatto(*args, xl1=xl1, xl2=xl2)
                              for args in zip(alpha, beta)])
        return coeffs[:, 0], coeffs[:, 1]

    alpha = numpy.array(alpha)
    beta = numpy.array(beta)
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
    return alpha, beta
