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
import chaospy

from ..baseclass import Distribution, J
from .operator import OperatorDistribution


class Trunc(Distribution):
    """Truncation."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Distribution, numpy.ndarray):
                Left hand side.
            right (Distribution, numpy.ndarray):
                Right hand side.
        """
        repr_args = [left, right]
        if isinstance(left, Distribution):
            if left.stochastic_dependent:
                raise chaospy.StochasticallyDependentError(
                    "Joint distribution with dependencies not supported.")
            exclusion = {dep for deps in left._dependencies for dep in deps}
        else:
            left = numpy.atleast_1d(left)
        if isinstance(right, Distribution):
            if right.stochastic_dependent:
                raise chaospy.StochasticallyDependentError(
                    "Joint distribution with dependencies not supported.")
            exclusion = {dep for deps in right._dependencies for dep in deps}
        else:
            right = numpy.atleast_1d(right)

        length = max(len(left), len(right))
        assert isinstance(left, Distribution) or isinstance(right, Distribution)
        dependencies = [
            set([idx]) for idx in self._declare_dependencies(length)]
        super(Trunc, self).__init__(
            parameters=dict(left=left, right=right),
            dependencies=dependencies,
            exclusion=exclusion,
            repr_args=repr_args,
        )

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
        if isinstance(left, Distribution):
            left = left._get_lower(cache=cache)
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
        if isinstance(right, Distribution):
            right = right._get_upper(cache=cache)
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
        if isinstance(left, Distribution):
            left = left._get_value(cache)
        if isinstance(right, Distribution):
            right = right._get_value(cache)
        if isinstance(left, Distribution):
            right = (numpy.array(right).T*numpy.ones(xloc.shape).T).T
            uloc1 = left._get_fwd(right, cache=cache.copy())
            uloc2 = left._get_fwd(xloc, cache=cache)
            out = uloc2/uloc1
        else:
            left = (numpy.array(left).T*numpy.ones(xloc.shape).T).T
            uloc1 = right._get_fwd(left, cache=cache.copy())
            uloc2 = right._get_fwd(xloc, cache=cache)
            out = (uloc2-uloc1)/(1-uloc1)
        return out

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
        if isinstance(left, Distribution):
            left = left._get_value(cache)
        if isinstance(right, Distribution):
            right = right._get_value(cache)
        if isinstance(left, Distribution):
            right = (numpy.array(right).T*numpy.ones(xloc.shape).T).T
            uloc1 = left._get_fwd(right, cache=cache.copy())
            uloc2 = left._get_pdf(xloc, cache=cache)
            out = uloc2/uloc1
        else:
            left = (numpy.array(left).T*numpy.ones(xloc.shape).T).T
            uloc1 = right._get_fwd(left, cache=cache.copy())
            uloc2 = right._get_pdf(xloc, cache=cache)
            out = uloc2/(1-uloc1)
        return out

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
        if isinstance(left, Distribution):
            left = left._get_value(cache)
        if isinstance(right, Distribution):
            right = right._get_value(cache)
        if isinstance(left, Distribution):
            right = (numpy.array(right).T*numpy.ones(q.shape).T).T
            uloc = left._get_fwd(right, cache=cache.copy())
            out = left._get_inv(q*uloc, cache=cache)
        else:
            left = (numpy.array(left).T*numpy.ones(q.shape).T).T
            uloc = right._get_fwd(left, cache=cache.copy())
            out = right._get_inv(q*(1-uloc)+uloc, cache=cache)
        return out

    def _value(self, **kwargs):
        raise chaospy.UnsupportedFeature(
            "%s: does not support value retrieval." % self)


def Trunk(left, right):
    return Trunc(left, right)

def trunk(left, right):
    return Trunc(left, right)
