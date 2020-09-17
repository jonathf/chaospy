"""
Multiplication of distributions.

Example usage
-------------

Distribution multiplied with a constant::

    >>> distribution = chaospy.Uniform(0, 1)*4
    >>> distribution
    Mul(Uniform(), 4)
    >>> print(numpy.around(distribution.sample(5), 4))
    [2.6144 0.46   3.8011 1.9288 3.4899]
    >>> print(numpy.around(distribution.fwd([1, 2, 3]), 4))
    [0.25 0.5  0.75]
    >>> print(numpy.around(distribution.inv(distribution.fwd([1, 2, 3])), 4))
    [1. 2. 3.]
    >>> print(numpy.around(distribution.pdf([1, 2, 3]), 4))
    [0.25 0.25 0.25]
    >>> print(numpy.around(distribution.mom([1, 2, 3]), 4))
    [ 2.      5.3333 16.    ]
    >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
    [[2.     2.     2.    ]
     [1.3333 1.0667 1.0286]]

Construct joint multiplication distribution::

    >>> lhs = chaospy.Uniform(-1, 0)
    >>> rhs = chaospy.Uniform(-3, -2)
    >>> multiplication = lhs * rhs
    >>> print(multiplication)
    Mul(Uniform(lower=-1, upper=0), Uniform(lower=-3, upper=-2))
    >>> joint1 = chaospy.J(lhs, multiplication)
    >>> print(joint1.lower)
    [-1. -0.]
    >>> print(joint1.upper)
    [0. 3.]
    >>> joint2 = chaospy.J(rhs, multiplication)
    >>> print(joint2.lower)
    [-3. -0.]
    >>> print(joint2.upper)
    [-2.  3.]
    >>> joint3 = chaospy.J(multiplication, lhs)
    >>> print(joint3.lower)
    [-0. -1.]
    >>> print(joint3.upper)
    [3. 0.]
    >>> joint4 = chaospy.J(multiplication, rhs)
    >>> print(joint4.lower)
    [-0. -3.]
    >>> print(joint4.upper)
    [ 3. -2.]

Generate random samples::

    >>> print(numpy.around(joint1.sample(4), 4))
    [[-0.7877 -0.9593 -0.6028 -0.7669]
     [ 2.2383  2.1172  1.6532  1.8345]]
    >>> print(numpy.around(joint2.sample(4), 4))
    [[-2.8177 -2.2565 -2.9304 -2.1147]
     [ 2.6843  2.1011  1.2174  0.0613]]

Forward transformations::

    >>> lcorr = numpy.array([-0.9, -0.5, -0.1])
    >>> rcorr = numpy.array([-2.99, -2.5, -2.01])
    >>> print(numpy.around(joint1.fwd([lcorr, lcorr*rcorr]), 4))
    [[0.1  0.5  0.9 ]
     [0.99 0.5  0.01]]
    >>> print(numpy.around(joint2.fwd([rcorr, lcorr*rcorr]), 4))
    [[0.01 0.5  0.99]
     [0.9  0.5  0.1 ]]

Inverse transformations::

    >>> print(numpy.around(joint1.inv(joint1.fwd([lcorr, lcorr*rcorr])), 4))
    [[-0.9   -0.5   -0.1  ]
     [ 2.691  1.25   0.201]]
    >>> print(numpy.around(joint2.inv(joint2.fwd([rcorr, lcorr*rcorr])), 4))
    [[-2.99  -2.5   -2.01 ]
     [ 2.691  1.25   0.201]]
"""
import numpy
import chaospy

from ..baseclass import Distribution
from .operator import OperatorDistribution


class Mul(OperatorDistribution):
    """Multiplication."""

    def __init__(self, left, right):
        """
        Args:
            left (Distribution, numpy.ndarray):
                Left hand side.
            right (Distribution, numpy.ndarray):
                Right hand side.
        """
        super(Mul, self).__init__(
            left=left,
            right=right,
            repr_args=[left, right],
        )

    def _lower(self, left, right, cache):
        """
        Distribution lower bounds.

        Example:
            >>> chaospy.Mul(chaospy.Uniform(-1, 2), -2).lower
            array([-4.])
            >>> chaospy.Mul(chaospy.Uniform(-1, 1), chaospy.Uniform(1, 2)).lower
            array([-2.])

        """
        # small hack to deal with sign-flipping boundaries.
        del cache
        left = self._parameters["left"]
        right = self._parameters["right"]
        if isinstance(left, Distribution):
            left_upper = left._get_upper(cache={})
            left_lower = left._get_lower(cache={})

            if isinstance(right, Distribution):
                right_upper = right._get_upper(cache={})
                right_lower = right._get_lower(cache={})

                out = numpy.min(numpy.broadcast_arrays(
                    left_lower.T*right_lower.T,
                    left_lower.T*right_upper.T,
                    left_upper.T*right_lower.T,
                    left_upper.T*right_upper.T,
                ), axis=0).T

            else:
                out = numpy.min([left_lower.T*right.T, left_upper.T*right.T], axis=0).T

        else:
            assert isinstance(right, Distribution)
            right_upper = right._get_upper(cache={})
            right_lower = right._get_lower(cache={})
            out = numpy.min([left.T*right_lower.T,
                             left.T*right_upper.T], axis=0).T

        return out

    def _upper(self, left, right, cache):
        """
        Distribution upper bounds.

        Example:
            >>> chaospy.Mul(chaospy.Uniform(-1, 2), -2).upper
            array([2.])
            >>> chaospy.Mul(chaospy.Uniform(-1, 1), chaospy.Uniform(1, 2)).upper
            array([2.])

        """
        # small hack to deal with sign-flipping boundaries.
        del cache
        left = self._parameters["left"]
        right = self._parameters["right"]
        if isinstance(left, Distribution):
            left_lower = left._get_lower(cache={})
            left_upper = left._get_upper(cache={})

            if isinstance(right, Distribution):
                right_lower = right._get_lower(cache={})
                right_upper = right._get_upper(cache={})

                out = numpy.max(numpy.broadcast_arrays(
                    (left_lower.T*right_lower.T).T,
                    (left_lower.T*right_upper.T).T,
                    (left_upper.T*right_lower.T).T,
                    (left_upper.T*right_upper.T).T,
                ), axis=0)

            else:
                out = numpy.max([left_lower*right, left_upper*right], axis=0)

        else:
            assert isinstance(right, Distribution)
            right_lower = right._get_lower(cache={})
            right_upper = right._get_upper(cache={})
            out = numpy.max([left*right_lower, left*right_upper], axis=0)

        return out

    def _cdf(self, xloc, left, right, cache):
        if isinstance(left, Distribution):
            left, right = right, left
        left = numpy.broadcast_arrays(left.T, xloc.T)[0].T
        valids = left != 0
        xloc_ = xloc.copy()
        xloc_.T[valids.T] = xloc.T[valids.T]/left.T[valids.T]
        uloc = right._get_fwd(xloc_, cache=cache)
        return numpy.where(left.T >= 0, uloc.T, 1-uloc.T).T

    def _ppf(self, uloc, left, right, cache):
        """
        Point percentile function.

        Example:
            >>> print(chaospy.Uniform().inv([0.1, 0.2, 0.9]))
            [0.1 0.2 0.9]
            >>> print(Mul(chaospy.Uniform(), 2).inv([0.1, 0.2, 0.9]))
            [0.2 0.4 1.8]
            >>> print(Mul(2, chaospy.Uniform()).inv([0.1, 0.2, 0.9]))
            [0.2 0.4 1.8]
            >>> dist = chaospy.Mul([2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> print(dist.inv([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]]))
            [[1.  1.2 1.4]
             [0.5 0.6 0.7]]
            >>> dist = chaospy.Mul(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> print(dist.inv([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]]))
            [[0.5 0.6 0.7]
             [1.  1.2 1.4]]

        """
        if isinstance(right, Distribution):
            left, right = right, left
        uloc = numpy.where(numpy.asfarray(right).T > 0, uloc.T, 1-uloc.T).T
        xloc = left._get_inv(uloc, cache=cache)
        xloc = (xloc.T*right.T).T

        return xloc

    def _pdf(self, xloc, left, right, cache):
        """
        Probability density function.

        Example:
            >>> print(chaospy.Uniform().pdf([-0.5, 0.5, 1.5, 2.5]))
            [0. 1. 0. 0.]
            >>> print(Mul(chaospy.Uniform(), 2).pdf([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.5 0.5 0. ]
            >>> print(Mul(2, chaospy.Uniform()).pdf([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.5 0.5 0. ]
            >>> dist = chaospy.Mul([2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> print(dist.pdf([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [0.5 0.5 0. ]
            >>> dist = chaospy.Mul(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> print(dist.pdf([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [0.5 0.5 0. ]

        """
        if isinstance(right, Distribution):
            left, right = right, left

        right = (numpy.asfarray(right).T+numpy.zeros(xloc.shape).T).T
        valids = right != 0
        xloc.T[valids.T] = xloc.T[valids.T]/right.T[valids.T]

        pdf = left._get_pdf(xloc, cache=cache)
        pdf.T[valids.T] /= right.T[valids.T]
        return numpy.abs(pdf)

    def _mom(self, key, left, right, cache):
        """
        Statistical moments.

        Example:
            >>> print(numpy.around(chaospy.Uniform().mom([0, 1, 2, 3]), 4))
            [1.     0.5    0.3333 0.25  ]
            >>> print(numpy.around(Mul(chaospy.Uniform(), 2).mom([0, 1, 2, 3]), 4))
            [1.     1.     1.3333 2.    ]
            >>> print(numpy.around(Mul(2, chaospy.Uniform()).mom([0, 1, 2, 3]), 4))
            [1.     1.     1.3333 2.    ]
            >>> print(numpy.around(Mul(chaospy.Uniform(), chaospy.Uniform()).mom([0, 1, 2, 3]), 4))
            [1.     0.25   0.1111 0.0625]
        """
        del cache
        if isinstance(left, Distribution):
            if left.shares_dependencies(right):
                raise chaospy.StochasticallyDependentError(
                    "product of dependent distributions not feasible: "
                    "{} and {}".format(left, right)
                )
            left = left._get_mom(key)
        else:
            left = (numpy.array(left).T**key).T
        if isinstance(right, Distribution):
            right = right._get_mom(key)
        else:
            right = (numpy.array(right).T**key).T
        return numpy.prod(left)*numpy.prod(right)

    def _ttr(self, kloc, left, right, cache):
        """Three terms recurrence coefficients."""
        del cache
        if isinstance(right, Distribution):
            if isinstance(left, Distribution):
                raise chaospy.StochasticallyDependentError(
                    "product of distributions not feasible: "
                    "{} and {}".format(left, right)
                )
            left, right = right, left
        coeff0, coeff1 = left._get_ttr(kloc)
        return coeff0*right, coeff1*right*right

    def _cache(self, left, right, cache):
        if isinstance(left, Distribution) or isinstance(right, Distribution):
            return self
        return left*right
