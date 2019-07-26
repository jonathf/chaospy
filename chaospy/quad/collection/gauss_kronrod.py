"""
Gauss-Kronrod quadrature is an adaptive method for Gaussian quadrature rule. It
builds on top of other quadrature rules by extending "pure" Gaussian quadrature
rules with extra abscissas and new weights such that already used abscissas can
be reused. For more details, see `Wikipedia article`_.

For each order ``N`` taken with ordinary Gaussian quadrature, Gauss-Kronrod
will create ``2N+1`` abscissas where all of the ``N`` "old" abscissas are all
interlaced between the "new" ones.

The algorithm is well suited for any Jacobi scheme, i.e. quadrature involving
Uniform or Beta distribution, and might work on others as well. However, it
will not work everywhere. For example `Kahaner and Monegato`_ showed that
higher order Gauss-Kronrod quadrature for Gauss-Hermite and Gauss-Laguerre does
not exist.

.. _Wikipedia article: https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula
.. _Kahaner and Monegato: https://link.springer.com/article/10.1007/BF01590820

Example usage
-------------

Generate Gauss-Kronrod quadrature rules::

    >>> distribution = chaospy.Beta(2, 2, lower=-1, upper=1)
    >>> for order in range(4):  # doctest: +NORMALIZE_WHITESPACE
    ...     X, W = chaospy.generate_quadrature(
    ...         order, distribution, rule="gauss_kronrod")
    ...     print("{} {}".format(numpy.around(X, 2), numpy.around(W, 2)))
    [[-0.65 -0.    0.65]] [0.23 0.53 0.23]
    [[-0.82 -0.45  0.    0.45  0.82]] [0.07 0.26 0.34 0.26 0.07]
    [[-0.89 -0.65 -0.34 -0.    0.34  0.65  0.89]]
     [0.03 0.12 0.22 0.26 0.22 0.12 0.03]
    [[-0.93 -0.77 -0.54 -0.29 -0.    0.29  0.54  0.77  0.93]]
     [0.01 0.06 0.13 0.19 0.22 0.19 0.13 0.06 0.01]

Compare Gauss-Kronrod builds on top of Gauss-Legendre quadrature to pure
Gauss-Legendre::

    >>> distribution = chaospy.Uniform(-1, 1)
    >>> for order in range(5):
    ...     Xl, Wl = chaospy.generate_quadrature(
    ...         order, distribution, rule="gaussian")
    ...     Xk, Wk = chaospy.generate_quadrature(
    ...         order, distribution, rule="gauss_kronrod")
    ...     print("{} {}".format(
    ...         numpy.around(Xl, 2), numpy.around(Xk[:, 1::2], 2)))
    [[0.]] [[-0.]]
    [[-0.58  0.58]] [[-0.58  0.58]]
    [[-0.77 -0.    0.77]] [[-0.77 -0.    0.77]]
    [[-0.86 -0.34  0.34  0.86]] [[-0.86 -0.34  0.34  0.86]]
    [[-0.91 -0.54  0.    0.54  0.91]] [[-0.91 -0.54 -0.    0.54  0.91]]

Gauss-Kronrod build on top of Gauss-Hermite quadrature::

    >>> distribution = chaospy.Normal(0, 1)
    >>> for order in range(2):
    ...     Xl, Wl = chaospy.generate_quadrature(
    ...         order, distribution, rule="gaussian")
    ...     Xk, Wk = chaospy.generate_quadrature(
    ...         order, distribution, rule="gauss_kronrod")
    ...     print("{} {}".format(numpy.around(Xl, 2), numpy.around(Xk, 2)))
    [[0.]] [[-1.73  0.    1.73]]
    [[-1.  1.]] [[-2.45 -1.    0.    1.    2.45]]

Applying Gauss-Kronrod to Gauss-Hermite quadrature, for an order known to not
exist::

    >>> chaospy.generate_quadrature(5, distribution, rule="gauss_kronrod")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    numpy.linalg.LinAlgError: \
Invalid recurrence coefficients can not be used for constructing Gaussian quadrature rule

Multivariate support::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Beta(4, 5))
    >>> X, W = chaospy.generate_quadrature(
    ...     1, distribution, rule="gauss_kronrod")
    >>> print(numpy.around(X, 3))
    [[0.037 0.037 0.037 0.037 0.037 0.211 0.211 0.211 0.211 0.211 0.5   0.5
      0.5   0.5   0.5   0.789 0.789 0.789 0.789 0.789 0.963 0.963 0.963 0.963
      0.963]
     [0.144 0.297 0.444 0.612 0.796 0.144 0.297 0.444 0.612 0.796 0.144 0.297
      0.444 0.612 0.796 0.144 0.297 0.444 0.612 0.796 0.144 0.297 0.444 0.612
      0.796]]
    >>> print(numpy.around(W, 3))
    [0.006 0.027 0.035 0.026 0.004 0.016 0.067 0.086 0.065 0.011 0.02  0.085
     0.11  0.083 0.014 0.016 0.067 0.086 0.065 0.011 0.006 0.027 0.035 0.026
     0.004]

Sources
-------

Code is adapted from `quadpy`_, which adapted his code from `W. Gautschi`_.
Algorithm for calculating Kronrod-Jacobi matrices was first published in paper
by `D. P. Laurie`_.

.. _quadpy: https://github.com/nschloe/quadpy
.. _W. Gautschi: https://www.cs.purdue.edu/archives/2002/wxg/codes/OPQ.html
.. _D. P. Laurie: https://doi.org/10.1090/S0025-5718-97-00861-2
"""
from __future__ import division
import math

import numpy

from ..recurrence import (
    construct_recurrence_coefficients, coefficients_to_quadrature)
from ..combine import combine


def quad_gauss_kronrod(
        order,
        dist,
        rule="F",
        accuracy=100,
        recurrence_algorithm="",
):
    """
    Generate the abscissas and weights in Gauss-Kronrod quadrature.

    Args:
        dist (chaospy.distributions.baseclass.Dist):
            The distribution which density will be used as weight function.
        order (int):
            The order of the quadrature.
        rule (str):
            In the case of ``lanczos`` or ``stieltjes``, defines the
            proxy-integration scheme.
        accuracy (int):
            In the case ``rule`` is used, defines the quadrature order of the
            scheme used. In practice, must be at least as large as ``order``.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights. If
            omitted, ``analytical`` will be tried first, and ``stieltjes`` used
            if that fails. See \
:func:`~chaospy.quad.recurrence.recurrence.construct_recurrence_coefficients`.

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
            Error raised if Kronrod algorithm results in negative recurrence
            coefficients.

    Example:
        >>> distribution = chaospy.Uniform(-1, 1)
        >>> abscissas, weights = quad_gauss_kronrod(6, distribution)
        >>> print(numpy.around(abscissas, 3))
        [[-0.991 -0.949 -0.865 -0.742 -0.586 -0.406 -0.208 -0.     0.208  0.406
           0.586  0.742  0.865  0.949  0.991]]
        >>> print(numpy.around(weights, 3))
        [0.011 0.032 0.052 0.07  0.085 0.095 0.102 0.105 0.102 0.095 0.085 0.07
         0.052 0.032 0.011]
    """
    # assert not rule.startswith("gauss"), "recursive Gaussian quadrature call"

    length = int(numpy.ceil(3*(order+1) / 2.0))
    coefficients = construct_recurrence_coefficients(
        length, dist, rule, accuracy, recurrence_algorithm)

    coefficients = [kronrod_jacobi(order+1, coeffs) for coeffs in coefficients]

    abscissas, weights = coefficients_to_quadrature(coefficients)

    abscissas = combine(abscissas).T.reshape(len(dist), -1)
    weights = numpy.prod(combine(weights), -1)

    return abscissas, weights


def kronrod_jacobi(order, coeffs):
    """
    Create the three-terms-recursion coefficients resulting from the
    Kronrod-Jacobi matrix.

    Augment three terms recurrence coefficients to add extra Gauss-Kronrod
    terms.

    Args:
        order (int):
            Order of the Gaussian quadrature rule.
        coeffs (numpy.ndarray):
            Three terms recurrence coefficients of the Gaussian quadrature
            rule.

    Returns:
        Three terms recurrence coefficients of the Gauss-Kronrod quadrature
        rule.
    """
    assert len(coeffs[0]) == int(math.ceil(3*order/2.0))+1

    bound = int(math.floor(3*order/2.0))+1
    coeffs_a = numpy.zeros(2*order+1)
    coeffs_a[:bound] = coeffs[0][:bound]

    bound = int(math.ceil(3*order/2.0))+1
    coeffs_b = numpy.zeros(2*order+1)
    coeffs_b[:bound] = coeffs[1][:bound]

    sigma = numpy.zeros((2, order//2+2))
    sigma[1, 1] = coeffs_b[order+1]

    for idx in range(order-1):
        idy = numpy.arange((idx+1)//2, -1, -1)
        sigma[0, idy+1] = numpy.cumsum(
            (coeffs_a[idy+order+1]-coeffs_a[idx-idy])*sigma[1, idy+1]+
            coeffs_b[idy+order+1]*sigma[0, idy]-
            coeffs_b[idx-idy]*sigma[0, idy+1]
        )
        sigma = numpy.roll(sigma, 1, axis=0)

    sigma[0, 1:order//2+2] = sigma[0, :order//2+1]
    for idx in range(order-1, 2*order-2):
        idy = numpy.arange(idx-order+1, (idx-1)//2+1)
        j = order-1-idx+idy
        sigma[0, j+1] = numpy.cumsum(
            -(coeffs_a[idy+order+1]-coeffs_a[idx-idy])*sigma[1, j+1]-
            coeffs_b[idy+order+1]*sigma[0, j+1]+
            coeffs_b[idx-idy]*sigma[0, j+2]
        )
        j = j[-1]
        idy = (idx+1)//2
        if idx%2 == 0:
            coeffs_a[idy+order+1] = (
                coeffs_a[idy]+
                (sigma[0, j+1]-coeffs_b[idy+order+1]*sigma[0, j+2])/
                sigma[1, j+2]
            )
        else:
            coeffs_b[idy+order+1] = sigma[0, j+1]/sigma[0, j+2]
        sigma = numpy.roll(sigma, 1, axis=0)

    coeffs_a[2*order] = (coeffs_a[order-1]-
                         coeffs_b[2*order]*sigma[0, 1]/sigma[1, 1])
    return numpy.asfarray([coeffs_a, coeffs_b])
