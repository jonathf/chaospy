"""
Addition operator.

Example usage
-------------

Distribution and a constant::

    >>> distribution = chaospy.Normal(0, 1) + 10
    >>> distribution
    Add(Normal(mu=0, sigma=1), 10)
    >>> distribution.sample(5).round(4)
    array([10.395 ,  8.7997, 11.6476,  9.9553, 11.1382])
    >>> distribution.fwd([9, 10, 11]).round(4)
    array([0.1587, 0.5   , 0.8413])
    >>> distribution.inv(distribution.fwd([9, 10, 11])).round(4)
    array([ 9., 10., 11.])
    >>> distribution.pdf([9, 10, 11]).round(4)
    array([0.242 , 0.3989, 0.242 ])
    >>> distribution.mom([1, 2, 3]).round(4)
    array([  10.,  101., 1030.])
    >>> distribution.ttr([1, 2, 3]).round(4)
    array([[10., 10., 10.],
           [ 1.,  2.,  3.]])

Construct joint addition distribution::

    >>> lhs = chaospy.Uniform(2, 3)
    >>> rhs = chaospy.Uniform(3, 4)
    >>> addition = lhs + rhs
    >>> addition
    Add(Uniform(lower=2, upper=3), Uniform(lower=3, upper=4))
    >>> joint1 = chaospy.J(lhs, addition)
    >>> joint2 = chaospy.J(rhs, addition)

Generate random samples::

    >>> joint1.sample(4).round(4)
    array([[2.2123, 2.0407, 2.3972, 2.2331],
           [6.0541, 5.2478, 6.1397, 5.6253]])
    >>> joint2.sample(4).round(4)
    array([[3.1823, 3.7435, 3.0696, 3.8853],
           [6.1349, 6.6747, 5.485 , 5.9143]])

Forward transformations::

    >>> lcorr = numpy.array([2.1, 2.5, 2.9])
    >>> rcorr = numpy.array([3.01, 3.5, 3.99])
    >>> joint1.fwd([lcorr, lcorr+rcorr]).round(4)
    array([[0.1 , 0.5 , 0.9 ],
           [0.01, 0.5 , 0.99]])
    >>> joint2.fwd([rcorr, lcorr+rcorr]).round(4)
    array([[0.01, 0.5 , 0.99],
           [0.1 , 0.5 , 0.9 ]])

Inverse transformations::

    >>> joint1.inv(joint1.fwd([lcorr, lcorr+rcorr])).round(4)
    array([[2.1 , 2.5 , 2.9 ],
           [5.11, 6.  , 6.89]])
    >>> joint2.inv(joint2.fwd([rcorr, lcorr+rcorr])).round(4)
    array([[3.01, 3.5 , 3.99],
           [5.11, 6.  , 6.89]])

Raw moments::

    >>> joint1.mom([(0, 1, 1), (1, 0, 1)]).round(4)
    array([ 6.0006,  2.5002, 15.0847])
    >>> joint2.mom([(0, 1, 1), (1, 0, 1)]).round(4)
    array([ 6.0006,  3.5003, 21.0853])
"""
from __future__ import division
from scipy.special import comb
import numpy

from ..baseclass import Dist, StochasticallyDependentError
from .. import evaluation
from .binary import BinaryOperator


class Add(BinaryOperator):
    """Addition."""

    def _lower(self, left, right, cache):
        """
        Distribution bounds.

        Example:
            >>> chaospy.Uniform().lower
            array([0.])
            >>> chaospy.Add(chaospy.Uniform(), 2).lower
            array([2.])
            >>> chaospy.Add(2, chaospy.Uniform()).lower
            array([2.])
            >>> chaospy.Add(1, 1).lower
            array([2.])
        """
        left = evaluation.get_forward_cache(left, cache)
        right = evaluation.get_forward_cache(right, cache)
        if isinstance(left, Dist):
            left = evaluation.evaluate_lower(left, cache=cache)
        if isinstance(right, Dist):
            right = evaluation.evaluate_lower(right, cache=cache)
        return left+right

    def _upper(self, left, right, cache):
        """
        Distribution bounds.

        Example:
            >>> chaospy.Uniform().upper
            array([1.])
            >>> chaospy.Add(chaospy.Uniform(), 2).upper
            array([3.])
            >>> chaospy.Add(2, chaospy.Uniform()).upper
            array([3.])
            >>> chaospy.Add(1, 1).upper
            array([2.])
        """
        left = evaluation.get_forward_cache(left, cache)
        right = evaluation.get_forward_cache(right, cache)
        if isinstance(left, Dist):
            left = evaluation.evaluate_upper(left, cache=cache)
        if isinstance(right, Dist):
            right = evaluation.evaluate_upper(right, cache=cache)
        return left+right

    def _pre_fwd_left(self, xloc, other):
        xloc = (xloc.T-numpy.asfarray(other).T).T
        return xloc

    def _pre_fwd_right(self, xloc, other):
        xloc = (xloc.T-numpy.asfarray(other).T).T
        return xloc

    def _post_fwd(self, uloc, other):
        del other
        return uloc

    def _alt_fwd(self, xloc, left, right):
        return numpy.asfarray(left+right <= xloc)

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
        left = evaluation.get_forward_cache(left, cache)
        right = evaluation.get_forward_cache(right, cache)

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))
        elif not isinstance(right, Dist):
            return numpy.inf
        else:
            left, right = right, left

        xloc = (xloc.T-numpy.asfarray(right).T).T
        output = evaluation.evaluate_density(left, xloc, cache=cache)
        assert output.shape == xloc.shape
        return output

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
        left = evaluation.get_inverse_cache(left, cache)
        right = evaluation.get_inverse_cache(right, cache)

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))
        elif not isinstance(right, Dist):
            return left+right
        else:
            left, right = right, left

        xloc = evaluation.evaluate_inverse(left, uloc, cache=cache)
        output = (xloc.T + numpy.asfarray(right).T).T
        return output

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
                evaluation.evaluate_moment(left, key, cache=cache)
                for key in keys_.T
            ]
        else:
            left = list(reversed(numpy.array(left).T**keys_.T))
        if isinstance(right, Dist):
            right = [
                evaluation.evaluate_moment(right, key, cache=cache)
                for key in keys_.T
            ]
        else:
            right = list(reversed(numpy.array(right).T**keys_.T))

        out = numpy.zeros(keys.shape)
        for idx in range(keys_.shape[1]):
            key = keys_.T[idx]
            coef = comb(keys.T, key)
            out += coef*left[idx]*right[idx]*(key <= keys.T)

        if numpy.asarray(out).shape:
            out = numpy.prod(out)
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
            >>> print(numpy.around(chaospy.Add(1, 1).ttr([0, 1, 2, 3]), 4)) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            chaospy.distributions.baseclass.StochasticallyDependentError: recurrence ...
        """
        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise StochasticallyDependentError(
                    "sum of distributions not feasible: "
                    "{} and {}".format(left, right)
                )
        else:
            if not isinstance(right, Dist):
                raise StochasticallyDependentError(
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
        return super(Add, self).__str__()

    def _fwd_cache(self, cache):
        left = evaluation.get_forward_cache(self.prm["left"], cache)
        right = evaluation.get_forward_cache(self.prm["right"], cache)
        if not isinstance(left, Dist) and not isinstance(right, Dist):
            return left+right
        return self

    def _inv_cache(self, cache):
        left = evaluation.get_forward_cache(self.prm["left"], cache)
        right = evaluation.get_forward_cache(self.prm["right"], cache)
        if not isinstance(left, Dist) and not isinstance(right, Dist):
            return left+right
        return self


def add(left, right):
    return Add(left, right)
