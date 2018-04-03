"""
Negative of a distribution.

Example usage
-------------

Invert sign of a distribution::

    >>> distribution = -chaospy.Uniform(0, 1)
    >>> print(distribution)
    Neg(Uniform(lower=0, upper=1))
    >>> print(numpy.around(distribution.sample(5), 4))
    [-0.3464 -0.885  -0.0497 -0.5178 -0.1275]
    >>> print(distribution.fwd([-0.3, -0.2, -0.1]))
    [0.7 0.8 0.9]
    >>> print(distribution.inv(distribution.fwd([-0.3, -0.2, -0.1])))
    [-0.3 -0.2 -0.1]
    >>> print(distribution.pdf([-0.3, -0.2, -0.1]))
    [1. 1. 1.]
    >>> print(numpy.around(distribution.mom([1, 2, 3]), 4))
    [-0.5     0.3333 -0.25  ]
    >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
    [[-0.5    -0.5    -0.5   ]
     [ 0.0833  0.0667  0.0643]]

"""
import numpy
from ..baseclass import Dist
from .. import evaluation


class Neg(Dist):
    """Negative of a distribution."""

    def __init__(self, dist):
        """
        Constructor.

        Args:
            dist (Dist) : distribution.
        """
        Dist.__init__(self, dist=dist)

    def _bnd(self, xloc, dist, cache):
        """Distribution bounds."""
        return -evaluation.evaluate_bound(dist, -xloc)[::-1]

    def _pdf(self, xloc, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(dist, -xloc, cache=cache)

    def _cdf(self, xloc, dist, cache):
        """Cumulative distribution function."""
        return 1-evaluation.evaluate_forward(dist, -xloc, cache=cache)

    def _ppf(self, q, dist, cache):
        """Point percentile function."""
        return -evaluation.evaluate_inverse(dist, 1-q, cache=cache)

    def _mom(self, k, dist, cache):
        """Statistical moments."""
        return (-1)**numpy.sum(k)*evaluation.evaluate_moment(
            dist, k, cache=cache)

    def _ttr(self, k, dist, cache):
        """Three terms recursion coefficients."""
        a,b = evaluation.evaluate_recurrence_coefficients(dist, k)
        return -a, b

    def __str__(self):
        """String representation."""
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"


def neg(left):
    """
    Negative of a distribution.

    Args:
        dist (Dist) : distribution.
    """
    if not isinstance(left, Dist):
        return -left
    return Neg(left)
