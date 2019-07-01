"""
Gauss-Kronrod quadrature; see
<https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula>.

The Jacobi matrix of the :math:`(2n+1)`-point Gauss-Kronrod quadrature rule for
a given measure is calculated efficiently by a five-term recurrence relation.
The algorithm uses only rational operations and is therefore also useful for
obtaining the Jacobi-Kronrod matrix analytically. The nodes and weights can
then be computed directly by standard software for Gaussian quadrature
formulas.

Code adapted from
<https://github.com/nschloe/quadpy/quadpy/line_segment/_gauss_kronrod.py>

Calculation of Gauss-Kronrod quadrature rules,
Dirk P. Laurie,
Math. Comp. 66 (1997), 1133-1145,
<https://doi.org/10.1090/S0025-5718-97-00861-2>

Example usage
-------------

The first few orders with linear growth rule::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> for order in [0, 1, 2, 3, 4]:
    ...     X, W = chaospy.generate_quadrature(order, distribution, rule="K")
    ...     print("{} {} {}".format(
    ...         order, numpy.around(X, 3), numpy.around(W, 3)))
    0 [[0.5]] [1.]
    1 [[0.211 0.789]] [0.5 0.5]
    2 [[0.113 0.5   0.887]] [0.278 0.444 0.278]
    3 [[0.069 0.33  0.67  0.931]] [0.174 0.326 0.326 0.174]
    4 [[0.047 0.231 0.5   0.769 0.953]] [0.118 0.239 0.284 0.239 0.118]
"""
import math

import numpy

from .golub_welsch import _golbub_welsch
from ..stieltjes import generate_stieltjes


def quad_gauss_kronrod(order, lower=0., upper=1., growth=False):
    """
    Generate the quadrature nodes and weights in Gauss-Kronrod quadrature.

    Example:
        >>> abscissas, weights = quad_gauss_kronrod(3, lower=-1, upper=1)
        >>> print(numpy.around(abscissas, 4))
        [[-0.8611 -0.34    0.34    0.8611]]
        >>> print(numpy.around(weights, 4))
        [0.1739 0.3261 0.3261 0.1739]
    """
    # The general scheme is:
    from chaospy.distributions.collection import Beta

    # Get the Jacobi recurrence coefficients
    length = int(math.ceil(3 * (order+1) / 2.0))
    _, _, coeffs_a, coeffs_b = generate_stieltjes(
        Beta(1, 1, lower=-1, upper=1), length, retall=True)

    # get the Kronrod vectors alpha and beta
    alpha, beta = _r_kronrod(order+1, coeffs_a, coeffs_b)

    # Solve eigen problem for a tridiagonal matrix with alpha and beta
    abscisas, weight = _golbub_welsch([order+1], alpha, beta)

    # Scale to arbitrary interval
    abscisas = (numpy.asfarray(abscisas)*0.5+0.5)*(upper-lower) + lower

    return abscisas, numpy.asfarray(weight[0])


def _r_kronrod(order, a0, b0):
    assert len(a0[0]) == int(math.ceil(3*order/2.0))+1
    assert len(b0[0]) == int(math.ceil(3*order/2.0))+1

    a = numpy.zeros(2*order+1)
    b = numpy.zeros(2*order+1)

    k = int(math.floor(3*order/2.0))+1
    a[:k] = a0[0, :k]
    k = int(math.ceil(3*order/2.0))+1
    b[:k] = b0[0, :k]
    s = numpy.zeros(int(math.floor(order / 2.0)) + 2)
    t = numpy.zeros(int(math.floor(order / 2.0)) + 2)
    t[1] = b[order+1]

    for m in range(order-1):
        k0 = int(math.floor((m+1)/2.0))
        k = numpy.arange(k0, -1, -1)
        L = m - k
        s[k + 1] = numpy.cumsum(
            (a[k+order+1]-a[L])*t[k+1]+b[k+order+1]*s[k]-b[L]*s[k + 1])
        s, t = t, s

    j = int(math.floor(order/2.0))+1
    s[1:j+1] = s[:j]
    for m in range(order-1, 2*order-2):
        k0 = m+1-order
        k1 = int(math.floor((m-1)/2.0))
        k = numpy.arange(k0, k1+1)
        L = m-k
        j = order-1-L
        s[j+1] = numpy.cumsum(
            -(a[k+order+1]-a[L])*t[j+1]-b[k+order+1]*s[j+1]+b[L]*s[j+2])
        j = j[-1]
        k = int(math.floor((m+1)/2.0))
        if m%2 == 0:
            a[k+order+1] = a[k]+(s[j+1]-b[k+order+1]*s[j+2])/t[j+2]
        else:
            b[k+order+1] = s[j+1]/s[j+2]
        s, t = t, s

    a[2*order] = a[order-1]-b[2*order]*s[1]/t[1]
    return a.reshape(1, -1), b.reshape(1, -1)
