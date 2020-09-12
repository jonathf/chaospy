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
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="gauss_radau")
    ...     print(abscissas.round(2), weights.round(2))
    [[-1.]] [1.]
    [[-1.   0.2]] [0.17 0.83]
    [[-1.   -0.51  0.13  0.71]] [0.02 0.33 0.48 0.17]
    [[-1.   -0.74 -0.35  0.1   0.53  0.85]]
     [0.01 0.11 0.28 0.34 0.21 0.05]

Multivariate samples::

    >>> distribution = chaospy.J(
    ...     chaospy.Uniform(0, 1), chaospy.Beta(4, 5))
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     1, distribution, rule="gauss_radau")
    >>> abscissas.round(3)
    array([[0.   , 0.   , 0.667, 0.667],
           [0.   , 0.5  , 0.   , 0.5  ]])
    >>> weights.round(3)
    array([0.028, 0.222, 0.083, 0.667])

To change the fixed point, the direct generating function has to be used::

    >>> distribution = chaospy.Uniform(lower=-1, upper=1)
    >>> for fixed_point in numpy.linspace(-1, 1, 6):
    ...     abscissas, weights = chaospy.quad_gauss_radau(
    ...         2, distribution, fixed_point)
    ...     print(abscissas.round(2), weights.round(2))
    [[-1.   -0.58  0.18  0.82]] [0.06 0.33 0.39 0.22]
    [[-1.04 -0.6   0.17  0.82]] [0.05 0.33 0.39 0.22]
    [[-0.83 -0.2   0.54  0.96]] [0.22 0.38 0.32 0.08]
    [[-0.96 -0.54  0.2   0.83]] [0.08 0.32 0.38 0.22]
    [[-0.82 -0.17  0.6   1.04]] [0.22 0.39 0.33 0.05]
    [[-0.82 -0.18  0.58  1.  ]] [0.22 0.39 0.33 0.06]

However, a fixed point at 0 is not allowed::

    >>> chaospy.quad_gauss_radau(  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...     3, distribution, fixed_point=0)
    Traceback (most recent call last):
        ...
    numpy.linalg.LinAlgError: Illegal Radau fixed point: 0.0
"""
import numpy
import scipy.linalg

from .recurrence import (
    construct_recurrence_coefficients, coefficients_to_quadrature)
from .combine import combine_quadrature


def quad_gauss_radau(
        order,
        dist,
        fixed_point=None,
        rule="fejer",
        accuracy=100,
        recurrence_algorithm="",
):
    """
    Generate the quadrature nodes and weights in Gauss-Radau quadrature.

    Args:
        order (int):
            Quadrature order.
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution weights to be used to create higher order nodes
            from.
        fixed_point (float):
            Fixed point abscissas assumed to be included in the quadrature. If
            imitted, use distribution lower point ``dist.range()[0]``.
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
        >>> abscissas, weights = quad_gauss_radau(4, chaospy.Uniform(-1, 1))
        >>> abscissas.round(3)
        array([[-1.   , -0.887, -0.64 , -0.295,  0.094,  0.468,  0.771,  0.955]])
        >>> weights.round(3)
        array([0.016, 0.093, 0.152, 0.188, 0.196, 0.174, 0.125, 0.057])
    """
    assert not rule.startswith("gauss"), "recursive Gaussian quadrature call"
    if fixed_point is None:
        fixed_point = dist.lower
    else:
        fixed_point = numpy.ones(len(dist))*fixed_point

    if order == 0:
        return fixed_point.reshape(-1, 1), numpy.ones(1)

    coefficients = construct_recurrence_coefficients(
        2*order-1, dist, rule, accuracy, recurrence_algorithm)
    coefficients = [radau_jakobi(coeffs, point)
                    for point, coeffs in zip(fixed_point, coefficients)]

    abscissas, weights = coefficients_to_quadrature(coefficients)

    return combine_quadrature(abscissas, weights)


def radau_jakobi(coeffs, fixed_point):
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
        numpy.linalg.LinAlgError:
            Error raised if fixed point causes the algorithm to break down.
    """
    right_hand_side = numpy.zeros(len(coeffs[0])-1)
    right_hand_side[-1] = coeffs[1][-1]
    bands_a = numpy.vstack([numpy.sqrt(coeffs[1]), coeffs[0]-fixed_point])
    bands_j = numpy.vstack((bands_a[:, 0:-1], bands_a[0, 1:]))
    try:
        delta = scipy.linalg.solve_banded((1, 1), bands_j, right_hand_side)
    except numpy.linalg.LinAlgError:
        raise numpy.linalg.LinAlgError(
            "Illegal Radau fixed point: %s" % fixed_point.item())
    coeffs = coeffs.copy()
    coeffs[0, -1] = fixed_point+delta[-1]
    return coeffs
