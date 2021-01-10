"""
Generate Gauss-Kronrod quadrature abscissas and weights.

Example usage
-------------

Generate Gauss-Kronrod quadrature rules::

    >>> distribution = chaospy.Beta(2, 2, lower=-1, upper=1)
    >>> for order in range(5):  # doctest: +NORMALIZE_WHITESPACE
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="kronrod")
    ...     print(abscissas.round(2), weights.round(2))
    [[0.]] [1.]
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
    ...     abscissas1, weights1 = chaospy.generate_quadrature(
    ...         order, distribution, rule="gaussian")
    ...     abscissas2, weights2 = chaospy.generate_quadrature(
    ...         order+1, distribution, rule="kronrod")
    ...     print(abscissas1.round(2), abscissas2[:, 1::2].round(2))
    [[0.]] [[-0.]]
    [[-0.58  0.58]] [[-0.58  0.58]]
    [[-0.77 -0.    0.77]] [[-0.77 -0.    0.77]]
    [[-0.86 -0.34  0.34  0.86]] [[-0.86 -0.34  0.34  0.86]]
    [[-0.91 -0.54  0.    0.54  0.91]] [[-0.91 -0.54 -0.    0.54  0.91]]

Gauss-Kronrod build on top of Gauss-Hermite quadrature::

    >>> distribution = chaospy.Normal(0, 1)
    >>> for order in range(2):
    ...     abscissas1, weights1 = chaospy.generate_quadrature(
    ...         order, distribution, rule="gaussian")
    ...     abscissas2, weights2 = chaospy.generate_quadrature(
    ...         order+1, distribution, rule="kronrod")
    ...     print(abscissas1.round(2), abscissas2.round(2))
    [[0.]] [[-1.73  0.    1.73]]
    [[-1.  1.]] [[-2.45 -1.    0.    1.    2.45]]

Applying Gauss-Kronrod to Gauss-Hermite quadrature, for an order known to not
exist::

    >>> chaospy.generate_quadrature(  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...     5, distribution, rule="kronrod")
    Traceback (most recent call last):
        ...
    numpy.linalg.LinAlgError: \
Invalid recurrence coefficients can not be used for constructing Gaussian quadrature rule

Multivariate support::

    >>> distribution = chaospy.J(
    ...     chaospy.Uniform(0, 1), chaospy.Beta(4, 5))
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     2, distribution, rule="kronrod")
    >>> abscissas.round(3)
    array([[0.037, 0.037, 0.037, 0.037, 0.037, 0.211, 0.211, 0.211, 0.211,
            0.211, 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.789, 0.789, 0.789,
            0.789, 0.789, 0.963, 0.963, 0.963, 0.963, 0.963],
           [0.144, 0.297, 0.444, 0.612, 0.796, 0.144, 0.297, 0.444, 0.612,
            0.796, 0.144, 0.297, 0.444, 0.612, 0.796, 0.144, 0.297, 0.444,
            0.612, 0.796, 0.144, 0.297, 0.444, 0.612, 0.796]])
    >>> weights.round(3)
    array([0.006, 0.027, 0.035, 0.026, 0.004, 0.016, 0.067, 0.086, 0.065,
           0.011, 0.02 , 0.085, 0.11 , 0.083, 0.014, 0.016, 0.067, 0.086,
           0.065, 0.011, 0.006, 0.027, 0.035, 0.026, 0.004])
"""
from __future__ import division
import math

import numpy
import chaospy

from .utils import combine_quadrature


def kronrod(
        order,
        dist,
        recurrence_algorithm="stieltjes",
        rule="clenshaw_curtis",
        tolerance=1e-10,
        scaling=3,
        n_max=5000,
):
    """
    Generate Gauss-Kronrod quadrature abscissas and weights.

    Gauss-Kronrod quadrature is an adaptive method for Gaussian quadrature
    rule. It builds on top of other quadrature rules by extending "pure"
    Gaussian quadrature rules with extra abscissas and new weights such that
    already used abscissas can be reused. For more details, see `Wikipedia
    article
    <https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula>`_.

    For each order ``N`` taken with ordinary Gaussian quadrature, Gauss-Kronrod
    will create ``2N+1`` abscissas where all of the ``N`` "old" abscissas are
    all interlaced between the "new" ones.

    The algorithm is well suited for any Jacobi scheme, i.e. quadrature
    involving Uniform or Beta distribution, and might work on others as well.
    However, it will not work everywhere. For example `Kahaner and Monegato
    <https://link.springer.com/article/10.1007/BF01590820>`_ showed that higher
    order Gauss-Kronrod quadrature for Gauss-Hermite and Gauss-Laguerre does
    not exist.

    Args:
        order (int):
            The order of the quadrature.
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution which density will be used as weight function.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights. If
            omitted, ``analytical`` will be tried first, and ``stieltjes`` used
            if that fails.
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

    Notes:
        Code is adapted from `quadpy <https://github.com/nschloe/quadpy>`_,
        which adapted his code from `W. Gautschi
        <https://www.cs.purdue.edu/archives/2002/wxg/codes/OPQ.html>`_.
        Algorithm for calculating Kronrod-Jacobi matrices was first published
        Algorithm as proposed by Laurie :cite:`laurie_calculation_1997`.

    Example:
        >>> distribution = chaospy.Uniform(-1, 1)
        >>> abscissas, weights = chaospy.quadrature.kronrod(4, distribution)
        >>> abscissas.round(2)
        array([[-0.98, -0.86, -0.64, -0.34,  0.  ,  0.34,  0.64,  0.86,  0.98]])
        >>> weights.round(3)
        array([0.031, 0.085, 0.133, 0.163, 0.173, 0.163, 0.133, 0.085, 0.031])

    """
    length = int(numpy.ceil(3*(order+1) / 2.0))
    coefficients = chaospy.construct_recurrence_coefficients(
        order=length,
        dist=dist,
        recurrence_algorithm=recurrence_algorithm,
        rule=rule,
        tolerance=tolerance,
        scaling=scaling,
        n_max=n_max,
    )

    coefficients = [kronrod_jacobi(order, coeffs) for coeffs in coefficients]

    abscissas, weights = chaospy.coefficients_to_quadrature(coefficients)
    return combine_quadrature(abscissas, weights)


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
    if not order:
        return kronrod_jacobi(1, coeffs)[:, :1]

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
