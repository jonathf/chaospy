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
"""
import numpy

from .joint import J
from ..baseclass import (
    Dist, StochasticallyDependentError, declare_stochastic_dependencies)
from .. import evaluation


class Trunc(Dist):
    """Truncation."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Dist, numpy.ndarray):
                Left hand side.
            right (Dist, numpy.ndarray):
                Right hand side.
        """
        if isinstance(left, Dist):
            self._exclusion = {dep for deps in left._dependencies for dep in deps}
            if len(left) > 1 and (not isinstance(left, J) or
                                      evaluation.get_dependencies(*list(left.inverse_map))):
                raise StochasticallyDependentError(
                    "Joint distribution with dependencies not supported.")
        if isinstance(right, Dist):
            self._exclusion = {dep for deps in right._dependencies for dep in deps}
            if len(right) > 1 and (not isinstance(right, J) or
                                   evaluation.get_dependencies(*list(right.inverse_map))):
                raise StochasticallyDependentError(
                    "Joint distribution with dependencies not supported.")

        assert isinstance(left, Dist) or isinstance(right, Dist)
        self._dependencies = [set([idx])
                              for idx in declare_stochastic_dependencies(self, len(self._exclusion))]
        Dist.__init__(self, left=left, right=right)


    def _lower(self, left, right, cache):
        """
        Distribution lower bound.

        Examples:
            >>> print(chaospy.Trunc(chaospy.Uniform(), 0.6).lower)
            [0.]
            >>> print(chaospy.Trunc(0.6, chaospy.Uniform()).lower)
            [0.6]
        """
        del right
        if isinstance(left, Dist):
            left = evaluation.evaluate_lower(left, cache=cache)
        return left

    def _upper(self, left, right, cache):
        """
        Distribution lower bound.

        Examples:
            >>> print(chaospy.Trunc(chaospy.Uniform(), 0.6).upper)
            [0.6]
            >>> print(chaospy.Trunc(0.6, chaospy.Uniform()).upper)
            [1.]
        """
        del left
        if isinstance(right, Dist):
            right = evaluation.evaluate_upper(right, cache=cache)
        return right

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
                raise StochasticallyDependentError(
                    "under-defined distribution {} or {}".format(left, right))
        else:
            left = (numpy.array(left).T*numpy.ones(xloc.shape).T).T
            uloc1 = evaluation.evaluate_forward(right, left, cache=cache.copy())
            uloc2 = evaluation.evaluate_forward(right, xloc, cache=cache)
            return (uloc2-uloc1)/(1-uloc1)

        right = (numpy.array(right).T*numpy.ones(xloc.shape).T).T
        uloc1 = evaluation.evaluate_forward(left, right, cache=cache.copy())
        uloc2 = evaluation.evaluate_forward(left, xloc, cache=cache)
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
                raise StochasticallyDependentError(
                    "under-defined distribution {} or {}".format(left, right))
        else:
            left = (numpy.array(left).T*numpy.ones(xloc.shape).T).T
            uloc1 = evaluation.evaluate_forward(right, left, cache=cache.copy())
            uloc2 = evaluation.evaluate_density(right, xloc, cache=cache)
            return uloc2/(1-uloc1)

        right = (numpy.array(right).T*numpy.ones(xloc.shape).T).T
        uloc1 = evaluation.evaluate_forward(left, right, cache=cache.copy())
        uloc2 = evaluation.evaluate_density(left, xloc, cache=cache)
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
                raise StochasticallyDependentError(
                    "under-defined distribution {} or {}".format(left, right))

            right = (numpy.array(right).T*numpy.ones(q.shape).T).T
            uloc = evaluation.evaluate_forward(left, right, cache=cache.copy())
            out = evaluation.evaluate_inverse(left, q*uloc, cache=cache)

        elif not isinstance(right, Dist):
            raise StochasticallyDependentError(
                "truncated variable indirectly depends on underlying variable")
        else:
            left = (numpy.array(left).T*numpy.ones(q.shape).T).T
            uloc = evaluation.evaluate_forward(right, left)
            out = evaluation.evaluate_inverse(right, q*(1-uloc)+uloc, cache=cache)
        return out

    def __str__(self):
        return (self.__class__.__name__ + "(" + str(self.prm["left"]) +
                ", " + str(self.prm["right"]) + ")")

def Trunk(left, right):
    return Trunc(left, right)

def trunk(left, right):
    return Trunc(left, right)
