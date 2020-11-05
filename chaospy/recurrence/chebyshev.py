"""Modified Chebyshev algorithm."""
from __future__ import division

import numpy


def modified_chebyshev(moments):
    r"""
    Given the first 2N raw statistical moments, this method uses the modified
    Chebyshev algorithm for computing the associated recurrence coefficients.

    Args:
        moments (numpy.ndarray):
            Raw statistical moments from calculating the integrals
            :math:`\int x^k p(x) dx` for :math:`k=0,\dots,2N`.

    Examples:
        >>> dist = chaospy.Normal()
        >>> modified_chebyshev(dist.mom(numpy.arange(8)))
        array([[0., 0., 0., 0.],
               [1., 1., 2., 3.]])
        >>> dist = chaospy.Uniform(-1, 1)
        >>> modified_chebyshev(dist.mom(numpy.arange(8)))
        array([[0.        , 0.        , 0.        , 0.        ],
               [1.        , 0.33333333, 0.26666667, 0.25714286]])
    """
    moments = numpy.asfarray(moments).flatten()
    order = len(moments)
    assert order%2 == 0

    sigma = numpy.zeros((3, order))
    sigma[0] = moments
    coeffs = [(sigma[0, 1]/sigma[0, 0], sigma[0, 0])]

    for idx in range(1, order//2):
        sigma[idx%3, idx:order-idx] = (
            sigma[(idx-1)%3, idx+1:order-idx+1]-
            coeffs[idx-1][0]*sigma[(idx-1)%3, idx:order-idx]-
            coeffs[idx-1][1]*sigma[(idx-2)%3, idx:order-idx]
        )
        coeffs.append((
            (sigma[idx%3, idx+1]/sigma[idx%3, idx]
             -sigma[(idx-1)%3, idx]/sigma[(idx-1)%3, idx-1]),
            sigma[idx%3, idx]/sigma[(idx-1)%3, idx-1],
        ))

    coeffs = numpy.array(coeffs[:order//2]).reshape(-1, 2).T
    return coeffs
