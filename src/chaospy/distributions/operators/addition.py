"""
Addition operator.

Example usage
-------------

Distribution and a constant::

    >>> distribution = chaospy.Normal(0, 1) + 10
    >>> print(distribution)
    Add(Normal(mu=0, sigma=1), 10)
    >>> print(numpy.around(distribution.sample(5), 4))
    [10.395   8.7997 11.6476  9.9553 11.1382]
    >>> print(numpy.around(distribution.fwd([9, 10, 11]), 4))
    [0.1587 0.5    0.8413]
    >>> print(numpy.around(distribution.inv(distribution.fwd([9, 10, 11])), 4))
    [ 9. 10. 11.]
    >>> print(numpy.around(distribution.pdf([9, 10, 11]), 4))
    [0.242  0.3989 0.242 ]
    >>> print(distribution.mom([1, 2, 3]))
    [  10.  101. 1030.]
    >>> print(distribution.ttr([1, 2, 3]))
    [[10. 10. 10.]
     [ 1.  2.  3.]]

Construct joint addition distribution::

    >>> lhs = chaospy.Uniform(2, 3)
    >>> rhs = chaospy.Uniform(3, 4)
    >>> addition = lhs + rhs
    >>> print(addition)
    Add(Uniform(lower=2, upper=3), Uniform(lower=3, upper=4))
    >>> joint1 = chaospy.J(lhs, addition)
    >>> joint2 = chaospy.J(rhs, addition)

Generate random samples::

    >>> print(numpy.around(joint1.sample(4), 4))
    [[2.2123 2.0407 2.3972 2.2331]
     [6.0541 5.2478 6.1397 5.6253]]
    >>> print(numpy.around(joint2.sample(4), 4))
    [[3.1823 3.7435 3.0696 3.8853]
     [6.1349 6.6747 5.485  5.9143]]

Forward transformations::

    >>> lcorr = numpy.array([2.1, 2.5, 2.9])
    >>> rcorr = numpy.array([3.01, 3.5, 3.99])
    >>> print(numpy.around(joint1.fwd([lcorr, lcorr+rcorr]), 4))
    [[0.1  0.5  0.9 ]
     [0.01 0.5  0.99]]
    >>> print(numpy.around(joint2.fwd([rcorr, lcorr+rcorr]), 4))
    [[0.01 0.5  0.99]
     [0.1  0.5  0.9 ]]

Inverse transformations::

    >>> print(numpy.around(joint1.inv(joint1.fwd([lcorr, lcorr+rcorr])), 4))
    [[2.1  2.5  2.9 ]
     [5.11 6.   6.89]]
    >>> print(numpy.around(joint2.inv(joint2.fwd([rcorr, lcorr+rcorr])), 4))
    [[3.01 3.5  3.99]
     [5.11 6.   6.89]]

Raw moments::

    >>> print(numpy.around(joint1.mom([(0, 1, 1), (1, 0, 1)]), 4))
    [ 6.   2.5 15. ]
    >>> print(numpy.around(joint2.mom([(0, 1, 1), (1, 0, 1)]), 4))
    [ 6.   3.5 21. ]
"""
from __future__ import division
from scipy.misc import comb
import numpy

from ..baseclass import Dist
from .. import evaluation, deprecations


class Add(Dist):
    """Addition."""

    def __init__(self, left, right):
        """
        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        Dist.__init__(self, left=left, right=right)

    def _bnd(self, xloc, left, right, cache):
        """
        Distribution bounds.

        Example:
            >>> print(chaospy.Uniform().range([-2, 0, 2, 4]))
            [[0. 0. 0. 0.]
             [1. 1. 1. 1.]]
            >>> print(chaospy.Add(chaospy.Uniform(), 2).range([-2, 0, 2, 4]))
            [[2. 2. 2. 2.]
             [3. 3. 3. 3.]]
            >>> print(chaospy.Add(2, chaospy.Uniform()).range([-2, 0, 2, 4]))
            [[2. 2. 2. 2.]
             [3. 3. 3. 3.]]
            >>> print(chaospy.Add(1, 1).range([-2, 0, 2, 4]))
            [[2. 2. 2. 2.]
             [2. 2. 2. 2.]]
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
            return left+right, left+right
        else:
            left, right = right, left

        right = numpy.asfarray(right)
        xloc = (xloc.T-numpy.asfarray(right).T).T
        lower, upper = evaluation.evaluate_bound(left, xloc, cache)
        return (lower.T+right.T).T, (upper.T+right.T).T

    def _cdf(self, xloc, left, right, cache):
        """
        Cumulative distribution function.

        Example:
            >>> print(chaospy.Uniform().fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.5 1.  1. ]
            >>> print(chaospy.Add(chaospy.Uniform(), 1).fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.  0.5 1. ]
            >>> print(chaospy.Add(1, chaospy.Uniform()).fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.  0.5 1. ]
            >>> print(chaospy.Add(1, 1).fwd([-0.5, 0.5, 1.5, 2.5]))
            [0. 0. 0. 1.]
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
            return numpy.asfarray(left+right <= xloc)
        else:
            left, right = right, left
        xloc = (xloc.T-numpy.asfarray(right).T).T
        return evaluation.evaluate_forward(left, xloc, cache)

    def _pdf(self, xloc, left, right, cache):
        """
        Probability density function.

        Example:
            >>> print(chaospy.Uniform().pdf([-2, 0, 2, 4]))
            [0. 1. 0. 0.]
            >>> print(chaospy.Add(chaospy.Uniform(), 2).pdf([-2, 0, 2, 4]))
            [0. 0. 1. 0.]
            >>> print(chaospy.Add(2, chaospy.Uniform()).pdf([-2, 0, 2, 4]))
            [0. 0. 1. 0.]
            >>> print(chaospy.Add(1, 1).pdf([-2, 0, 2, 4])) # Dirac logic
            [ 0.  0. inf  0.]
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
            return numpy.inf
        else:
            left, right = right, left

        xloc = (xloc.T-numpy.asfarray(right).T).T
        return evaluation.evaluate_density(left, xloc, cache)

    def _ppf(self, uloc, left, right, cache):
        """
        Point percentile function.

        Example:
            >>> print(chaospy.Uniform().inv([0.1, 0.2, 0.9]))
            [0.1 0.2 0.9]
            >>> print(chaospy.Add(chaospy.Uniform(), 2).inv([0.1, 0.2, 0.9]))
            [2.1 2.2 2.9]
            >>> print(chaospy.Add(2, chaospy.Uniform()).inv([0.1, 0.2, 0.9]))
            [2.1 2.2 2.9]
            >>> print(chaospy.Add(1, 1).inv([0.1, 0.2, 0.9]))
            [2. 2. 2.]
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
            return left+right
        else:
            left, right = right, left

        xloc = evaluation.evaluate_inverse(left, uloc, cache)
        return (xloc.T + numpy.asfarray(right).T).T

    def _mom(self, keys, left, right, cache):
        """
        Statistical moments.

        Example:
            >>> print(numpy.around(chaospy.Uniform().mom([0, 1, 2, 3]), 4))
            [1.     0.5    0.3333 0.25  ]
            >>> print(numpy.around(chaospy.Add(chaospy.Uniform(), 2).mom([0, 1, 2, 3]), 4))
            [ 1.      2.5     6.3333 16.25  ]
            >>> print(numpy.around(chaospy.Add(2, chaospy.Uniform()).mom([0, 1, 2, 3]), 4))
            [ 1.      2.5     6.3333 16.25  ]
            >>> print(numpy.around(chaospy.Add(1, 1).mom([0, 1, 2, 3]), 4))
            [1. 2. 4. 8.]
        """
        if evaluation.get_dependencies(left, right):
            raise evaluation.DependencyError(
                "sum of dependent distributions not feasible: "
                "{} and {}".format(left, right)
            )

        keys_ = numpy.mgrid[tuple(slice(0, key+1, 1) for key in keys)]
        keys_ = keys_.reshape(len(self), -1)

        if isinstance(left, Dist):
            left = [
                evaluation.evaluate_moment(left, key, cache)
                for key in keys_.T
            ]
        else:
            left = list(reversed(numpy.array(left).T**keys_.T))
        if isinstance(right, Dist):
            right = [
                evaluation.evaluate_moment(right, key, cache)
                for key in keys_.T
            ]
        else:
            right = list(reversed(numpy.array(right).T**keys_.T))

        out = numpy.zeros(keys.shape)
        for idx in range(keys_.shape[1]):
            key = keys_.T[idx]
            coef = comb(keys.T, key)
            out += coef*left[idx]*right[idx]*(key <= keys.T)

        if len(self) > 1:
            out = numpy.prod(out, 1)
        return out

    def _ttr(self, kloc, left, right, cache):
        """
        Three terms recursion coefficients.

        Example:
            >>> print(numpy.around(chaospy.Uniform().ttr([0, 1, 2, 3]), 4))
            [[ 0.5     0.5     0.5     0.5   ]
             [-0.      0.0833  0.0667  0.0643]]
            >>> print(numpy.around(chaospy.Add(chaospy.Uniform(), 2).ttr([0, 1, 2, 3]), 4))
            [[ 2.5     2.5     2.5     2.5   ]
             [-0.      0.0833  0.0667  0.0643]]
            >>> print(numpy.around(chaospy.Add(2, chaospy.Uniform()).ttr([0, 1, 2, 3]), 4))
            [[ 2.5     2.5     2.5     2.5   ]
             [-0.      0.0833  0.0667  0.0643]]
            >>> print(numpy.around(chaospy.Add(1, 1).ttr([0, 1, 2, 3]), 4))
            Traceback (most recent call last):
                ...
            chaospy.distributions.evaluation.DependencyError: recurrence ...
        """
        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "sum of distributions not feasible: "
                    "{} and {}".format(left, right)
                )
        else:
            if not isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "recurrence coefficients for constants not feasible: "
                    "{}".format(left+right)
                )
            left, right = right, left

        coeff0, coeff1 = evaluation.evaluate_recurrence_coefficients(
            left, kloc, cache=cache)
        return coeff0 + numpy.asarray(right), coeff1

    def __str__(self):
        if self._repr is None:
            return (self.__class__.__name__ + "(" + str(self.prm["left"]) +
                    ", " + str(self.prm["right"]) + ")")
        return super().__str__()

@deprecations.deprecation_warning
def add(left, right):
    return Add(left, right)
