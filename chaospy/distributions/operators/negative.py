"""
Negative of a distribution.

Example usage
-------------

Invert sign of a distribution::

    >>> distribution = -chaospy.Uniform()
    >>> distribution
    Negative(Uniform())
    >>> distribution.sample(5).round(4)
    array([-0.3464, -0.885 , -0.0497, -0.5178, -0.1275])
    >>> distribution.fwd([-0.3, -0.2, -0.1])
    array([0.7, 0.8, 0.9])
    >>> distribution.inv(distribution.fwd([-0.3, -0.2, -0.1]))
    array([-0.3, -0.2, -0.1])
    >>> distribution.pdf([-0.3, -0.2, -0.1])
    array([1., 1., 1.])
    >>> distribution.mom([1, 2, 3]).round(4)
    array([-0.5   ,  0.3333, -0.25  ])
    >>> distribution.ttr([1, 2, 3]).round(4)
    array([[-0.5   , -0.5   , -0.5   ],
           [ 0.0833,  0.0667,  0.0643]])

"""
import numpy

from ..baseclass import Distribution, OperatorDistribution


class Negative(OperatorDistribution):
    """Negative of a distribution."""

    _operator = lambda self, left, right: -left

    def __init__(self, dist):
        """
        Constructor.

        Args:
            dist (Distribution) : distribution.
        """
        super(Negative, self).__init__(
            left=dist,
            right=0,
        )
        self._repr_args = [dist]

    def _lower(self, idx, left, right, cache):
        del right
        return -left._get_upper(idx, cache)

    def _upper(self, idx, left, right, cache):
        del right
        return -left._get_lower(idx, cache)

    def _pdf(self, xloc, idx, left, right, cache):
        del right
        return left._get_pdf(-xloc, idx, cache)

    def _cdf(self, xloc, idx, left, right, cache):
        del right
        return 1-left._get_fwd(-xloc, idx, cache)

    def _ppf(self, uloc, idx, left, right, cache):
        del right
        return -left._get_inv(1-uloc, idx, cache)

    def _mom(self, kloc, left, right, cache):
        """Statistical moments."""
        del right
        del cache
        return (-1)**numpy.sum(kloc)*left._get_mom(kloc)

    def _ttr(self, kloc, idx, left, right, cache):
        """Three terms recurrence coefficients."""
        del right
        del cache
        alpha, beta = left._get_ttr(kloc, idx)
        return -alpha, beta


def neg(left):
    """
    Negative of a distribution.

    Args:
        dist (Distribution) : distribution.
    """
    return Negative(left)
