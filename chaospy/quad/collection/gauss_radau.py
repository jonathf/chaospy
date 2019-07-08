"""
Gauss-Radau formula for numerical estimation of integrals. It requires
:math:`m+1` points and fits all Polynomials to degree :math:`2m`, so it
effectively fits exactly all Polynomials of degree :math:`2m-1`.

It allows for a single abscissas to be user defined, while the others are built
around this point.

Canonically, Radau is built around Legendre weight function with the fixed
point at the left end. Not all distributions/fixed point combinations allows
for the building of a quadrature scheme.

Example usage
-------------

With increasing order::

    >>> distribution = chaospy.Beta(2, 2, lower=-1, upper=1)
    >>> for order in range(4):  # doctest: +NORMALIZE_WHITESPACE
    ...     X, W = chaospy.generate_quadrature(order, distribution, rule="R")
    ...     print("{} {}".format(numpy.around(X, 2), numpy.around(W, 2)))
    [[-1.]] [1.]
    [[-1.   0.2]] [0.17 0.83]
    [[-1.   -0.51  0.13  0.71]] [0.02 0.33 0.48 0.17]
    [[-1.   -0.74 -0.35  0.1   0.53  0.85]] [0.01 0.11 0.28 0.34 0.21 0.05]

Multivariate samples::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Beta(4, 5))
    >>> X, W = chaospy.generate_quadrature(1, distribution, rule="R")
    >>> print(numpy.around(X, 3))
    [[0.    0.    0.667 0.667]
     [0.    0.5   0.    0.5  ]]
    >>> print(numpy.around(W, 3))
    [0.028 0.222 0.083 0.667]

To change the fixed point, the direct generating function has to be used::

    >>> distribution = chaospy.Uniform(lower=-1, upper=1)
    >>> for fixed_point in numpy.linspace(-1, 1, 6):
    ...     X, W = chaospy.quad_gauss_radau(3, distribution, fixed_point)
    ...     print("{} {}".format(numpy.around(X, 2), numpy.around(W, 2)))
    [[-1.   -0.8  -0.39  0.12  0.6   0.92]] [0.03 0.16 0.24 0.26 0.21 0.1 ]
    [[-0.92 -0.6  -0.12  0.4   0.82  1.02]] [0.1  0.21 0.26 0.25 0.16 0.02]
    [[-0.93 -0.64 -0.2   0.28  0.69  0.94]] [0.09 0.19 0.24 0.23 0.17 0.08]
    [[-0.94 -0.69 -0.28  0.2   0.64  0.93]] [0.08 0.17 0.23 0.24 0.19 0.09]
    [[-1.02 -0.82 -0.4   0.12  0.6   0.92]] [0.02 0.16 0.25 0.26 0.21 0.1 ]
    [[-0.92 -0.6  -0.12  0.39  0.8   1.  ]] [0.1  0.21 0.26 0.24 0.16 0.03]

However, a fixed point at 0 is not allowed::

    >>> chaospy.quad_gauss_radau(3, distribution, fixed_point=0)
    Traceback (most recent call last):
        ...
    ValueError: Radau algorithm received illegal fixed points: [0.]
"""
import numpy
import scipy.linalg

from .golub_welsch import _golub_welsch
from ..stieltjes import generate_stieltjes
from ..combine import combine


def quad_gauss_radau(order, dist=None, fixed_point=None):
    """
    Generate the quadrature nodes and weights in Gauss-Radau quadrature.

    Args:
        order (int):
            Quadrature order.
        dist (chaospy.distributions.baseclass.Dist):
            The distribution weights to be used to create higher order nodes
            from. If omitted, use ``Uniform(-1, 1)``.
        fixed_point (float):
            Fixed point abscissas assumed to be included in the quadrature. If
            imitted, use distribution lower point ``dist.range()[0]``.

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
            Error raised if Radau algorithm fails to find recurrence
            coefficients.

    Example:
        >>> abscissas, weights = quad_gauss_radau(4)
        >>> print(numpy.around(abscissas, 3))
        [[-1.    -0.887 -0.64  -0.295  0.094  0.468  0.771  0.955]]
        >>> print(numpy.around(weights, 3))
        [0.016 0.093 0.152 0.188 0.196 0.174 0.125 0.057]
        >>> abscissas, weights = quad_gauss_radau(4, fixed_point=0)
        Traceback (most recent call last):
            ...
        ValueError: Radau algorithm received illegal fixed points: [0.]
    """
    if dist is None:
        from chaospy.distributions.collection import Uniform
        dist = Uniform(lower=-1, upper=1)

    if fixed_point is None:
        fixed_point, _ = dist.range()
    else:
        fixed_point = numpy.ones(len(dist))*fixed_point

    if order == 0:
        return fixed_point.reshape(-1, 1), numpy.ones(1)

    _, _, coeffs_a, coeffs_b = generate_stieltjes(dist, 2*order-1, retall=True)
    try:
        results = numpy.array([radau_jakobi(*coeffs) for coeffs in zip(
            coeffs_a, coeffs_b, fixed_point)])
    except numpy.linalg.LinAlgError:
        raise ValueError(
            "Radau algorithm received illegal fixed points: %s" % fixed_point)

    coeffs_a, coeffs_b = results[:, 0], results[:, 1]

    if numpy.any(coeffs_b < 0):
        raise ValueError(
            "Radau algorithm results in illegal coefficients;\n"
            "Gauss-Radau might not be  possible for %s" % dist
        )

    # Solve eigen problem for a tridiagonal matrix with As and Bs
    abscissas, weights = _golub_welsch(
        [len(coeffs_a[0])]*len(dist), coeffs_a, coeffs_b)
    abscissas = combine(abscissas).T
    weights = numpy.prod(combine(weights), -1)
    return abscissas, weights


def radau_jakobi(coeffs_a, coeffs_b, fixed_point):
    """
    Compute the Radau coefficients.

    Based on the section 7 of the paper "Some modified matrix eigenvalue
    problems", G. Golub.

    Args:
        coeffs_a (numpy.ndarray):
            The first three terms recurrence coefficients of the Gaussian
            quadrature rule.
        coeffs_b (numpy.ndarray):
            The second three terms recurrence coefficients of the Gaussian
            quadrature rule.
        fixed_point (float):
            Fixed point abscissas assumed to be included in the quadrature.

    Returns:
        Three terms recurrence coefficients of the Gauss-Radau quadrature rule.

    Raises:
        scipy.linalg.LinAlgError:
            Error raised if fixed point causes the algorithm to break down.
    """
    right_hand_side = numpy.zeros(len(coeffs_a)-1)
    right_hand_side[-1] = coeffs_b[-1]
    bands_a = numpy.vstack([numpy.sqrt(coeffs_b), coeffs_a-fixed_point])
    bands_j = numpy.vstack((bands_a[:, 0:-1], bands_a[0, 1:]))
    delta = scipy.linalg.solve_banded((1, 1), bands_j, right_hand_side)
    coeffs_a = coeffs_a.copy()
    coeffs_a[-1] = fixed_point+delta[-1]
    return coeffs_a, coeffs_b.copy()
