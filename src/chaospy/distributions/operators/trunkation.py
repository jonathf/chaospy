"""
Truncation.

Example usage
-------------

    >>> distribution = chaospy.Normal(0, 1)
    >>> print(numpy.around(distribution.inv([0.9, 0.99, 0.999]), 4))
    [1.2816 2.3263 3.0902]
    >>> distribution = chaospy.Normal(0, 1) < 1
    >>> print(numpy.around(distribution.inv([0.9, 0.99, 0.999]), 4))
    [0.6974 0.9658 0.9965]
    >>> distribution = chaospy.Normal(0, 1) < 2
    >>> print(numpy.around(distribution.inv([0.9, 0.99, 0.999]), 4))
    [1.1726 1.8449 1.9822]

Illegal dependencies:

    >>> dist1 = chaospy.Uniform()
    >>> dist2 = chaospy.Trunc(dist1, 0.5)
    >>> dist = chaospy.J(dist1, dist2)
    >>> dist.sample()
    Traceback (most recent call last):
        ...
    chaospy.distributions.evaluation.DependencyError: truncated variable ...
"""
import numpy

from .joint import J
from ..baseclass import Dist
from .. import evaluation, deprecations


class Trunc(Dist):
    """Truncation."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        if isinstance(left, Dist) and len(left) > 1:
            if (not isinstance(left, J) or
                    evaluation.get_dependencies(*list(left.inverse_map))):
                raise evaluation.DependencyError(
                    "Joint distribution with dependencies not supported.")
        if isinstance(right, Dist) and len(right) > 1:
            if (not isinstance(right, J) or
                    evaluation.get_dependencies(*list(right.inverse_map))):
                raise evaluation.DependencyError(
                    "Joint distribution with dependencies not supported.")

        assert isinstance(left, Dist) or isinstance(right, Dist)
        Dist.__init__(self, left=left, right=right)

    def _bnd(self, xloc, left, right, cache):
        """
        Distribution bounds.

        Example:
            >>> print(chaospy.Uniform().range([-2, 0, 2, 4]))
            [[0. 0. 0. 0.]
             [1. 1. 1. 1.]]
            >>> print(chaospy.Trunc(chaospy.Uniform(), 0.6).range([-2, 0, 2, 4]))
            [[0.  0.  0.  0. ]
             [0.6 0.6 0.6 0.6]]
            >>> print(chaospy.Trunc(0.4, chaospy.Uniform()).range([-2, 0, 2, 4]))
            [[0.4 0.4 0.4 0.4]
             [1.  1.  1.  1. ]]
        """
        if isinstance(left, Dist):
            if left in cache:
                left = cache[left]
            else:
                left = evaluation.evaluate_bound(left, xloc, cache)
        else:
            left = (numpy.array(left).T * numpy.ones((2,)+xloc.shape).T).T

        if isinstance(right, Dist):
            if right in cache:
                right = cache[right]
            else:
                right = evaluation.evaluate_bound(right, xloc, cache)
        else:
            right = (numpy.array(right).T * numpy.ones((2,)+xloc.shape).T).T

        return left[0], right[1]

    def _cdf(self, xloc, left, right, cache):
        """
        Cumulative distribution function.

        Example:
            >>> print(chaospy.Uniform().fwd([-0.5, 0.3, 0.7, 1.2]))
            [0.  0.3 0.7 1. ]
            >>> print(chaospy.Trunc(chaospy.Uniform(), 0.4).fwd([-0.5, 0.2, 0.8, 1.2]))
            [0.  0.5 1.  1. ]
            >>> print(chaospy.Trunc(0.6, chaospy.Uniform()).fwd([-0.5, 0.2, 0.8, 1.2]))
            [0.  0.  0.5 1. ]
        """
        if isinstance(left, Dist) and left in cache:
            left = cache[left]
        if isinstance(right, Dist) and right in cache:
            right = cache[right]

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))
        else:
            left = (numpy.array(left).T*numpy.ones(xloc.shape).T).T
            uloc1 = evaluation.evaluate_forward(right, left, cache.copy())
            uloc2 = evaluation.evaluate_forward(right, xloc, cache)
            return (uloc2-uloc1)/(1-uloc1)

        right = (numpy.array(right).T*numpy.ones(xloc.shape).T).T
        uloc1 = evaluation.evaluate_forward(left, right, cache.copy())
        uloc2 = evaluation.evaluate_forward(left, xloc, cache)
        return uloc2/uloc1

    def _pdf(self, xloc, left, right, cache):
        """
        Probability density function.

        Example:
            >>> dist = chaospy.Trunc(chaospy.Uniform(), 0.6)
            >>> print(dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25]))
            [0.         1.66666667 1.66666667 0.         0.        ]
            >>> dist = chaospy.Trunc(chaospy.Uniform(), 0.4)
            >>> print(dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25]))
            [0.  2.5 0.  0.  0. ]
            >>> dist = chaospy.Trunc(0.4, chaospy.Uniform())
            >>> print(dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25]))
            [0.         0.         1.66666667 1.66666667 0.        ]
            >>> dist = chaospy.Trunc(0.6, chaospy.Uniform())
            >>> print(dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25]))
            [0.  0.  0.  2.5 0. ]
        """
        if isinstance(left, Dist) and left in cache:
            left = cache[left]
        if isinstance(right, Dist) and right in cache:
            right = cache[right]

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))
        else:
            left = (numpy.array(left).T*numpy.ones(xloc.shape).T).T
            uloc1 = evaluation.evaluate_forward(right, left, cache.copy())
            uloc2 = evaluation.evaluate_density(right, xloc, cache)
            return uloc2/(1-uloc1)

        right = (numpy.array(right).T*numpy.ones(xloc.shape).T).T
        uloc1 = evaluation.evaluate_forward(left, right, cache.copy())
        uloc2 = evaluation.evaluate_density(left, xloc, cache)
        return uloc2/uloc1


    def _ppf(self, q, left, right, cache):
        """
        Point percentile function.

        Example:
            >>> print(chaospy.Uniform().inv([0.1, 0.2, 0.9]))
            [0.1 0.2 0.9]
            >>> print(chaospy.Trunc(chaospy.Uniform(), 0.4).inv([0.1, 0.2, 0.9]))
            [0.04 0.08 0.36]
            >>> print(chaospy.Trunc(0.6, chaospy.Uniform()).inv([0.1, 0.2, 0.9]))
            [0.64 0.68 0.96]
        """
        if isinstance(left, Dist) and left in cache:
            left = cache[left]
        if isinstance(right, Dist) and right in cache:
            right = cache[right]

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))
        elif not isinstance(right, Dist):
            raise evaluation.DependencyError(
                "truncated variable indirectly depends on underlying variable")
        else:
            left = (numpy.array(left).T*numpy.ones(q.shape).T).T
            uloc = evaluation.evaluate_forward(right, left)
            return evaluation.evaluate_inverse(right, q*(1-uloc)+uloc, cache)

        right = (numpy.array(right).T*numpy.ones(q.shape).T).T
        uloc = evaluation.evaluate_forward(left, right, cache.copy())
        return evaluation.evaluate_inverse(left, q*uloc, cache)

    def __str__(self):
        return (self.__class__.__name__ + "(" + str(self.prm["left"]) +
                ", " + str(self.prm["right"]) + ")")

@deprecations.deprecation_warning
def Trunk(left, right):
    return Trunc(left, right)

@deprecations.deprecation_warning
def trunk(left, right):
    return Trunc(left, right)
