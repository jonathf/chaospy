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
from .unary import UnaryOperator


class Neg(UnaryOperator):
    """Negative of a distribution."""

    def __init__(self, dist):
        """
        Constructor.

        Args:
            dist (Dist) : distribution.
        """
        Dist.__init__(self, dist=dist)
        self._repr = {"_": [dist]}

    def _post_pdf(self, xloc):
        return 1.

    def _pre_fwd(self, xloc):
        return -xloc

    def _post_fwd(self, uloc):
        return 1-uloc

    def _pre_inv(self, qloc):
        return 1-qloc

    def _post_inv(self, uloc):
        return -uloc

    def _lower(self, dist, cache, **kwargs):
        uloc = evaluation.evaluate_upper(dist, cache=cache)
        return self._post_inv(uloc, **kwargs)

    def _upper(self, dist, cache, **kwargs):
        uloc = evaluation.evaluate_lower(dist, cache=cache)
        return self._post_inv(uloc, **kwargs)

    def _mom(self, k, dist, cache):
        """Statistical moments."""
        return (-1)**numpy.sum(k)*evaluation.evaluate_moment(
            dist, k, cache=cache)

    def _ttr(self, k, dist, cache):
        """Three terms recursion coefficients."""
        a,b = evaluation.evaluate_recurrence_coefficients(dist, k)
        return -a, b


def neg(left):
    """
    Negative of a distribution.

    Args:
        dist (Dist) : distribution.
    """
    return Neg(left)
