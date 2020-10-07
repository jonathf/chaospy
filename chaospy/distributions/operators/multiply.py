"""
Multiplication of distributions.

Example usage
-------------

Distribution multiplied with a constant::

    >>> distribution = chaospy.Uniform(0, 1)*4
    >>> distribution
    Multiply(Uniform(), 4)
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
    Multiply(Uniform(lower=-1, upper=0), Uniform(lower=-3, upper=-2))
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

from ..baseclass import Distribution, OperatorDistribution


class Multiply(OperatorDistribution):
    """Multiplication."""

    def __init__(self, left, right):
        """
        Args:
            left (Distribution, numpy.ndarray):
                Left hand side.
            right (Distribution, numpy.ndarray):
                Right hand side.
        """
        self._cache_upper = {}
        self._cache_lower = {}
        super(Multiply, self).__init__(
            left=left,
            right=right,
            repr_args=[left, right],
        )

    def _lower(self, idx, left, right, cache):
        """
        Distribution lower bounds.

        Example:
            >>> chaospy.Multiply(chaospy.Uniform(-1, 2), -2).lower
            array([-4.])
            >>> chaospy.Multiply(chaospy.Uniform(-1, 1), chaospy.Uniform(1, 2)).lower
            array([-2.])

        """
        left = self._parameters["left"]
        right = self._parameters["right"]

        if isinstance(left, Distribution):
            left_upper = left._get_upper(idx, cache=self._upper_cache)
            left_lower = left._get_lower(idx, cache=self._lower_cache)

            if isinstance(right, Distribution):
                right_upper = right._get_upper(idx, cache=self._upper_cache)
                right_lower = right._get_lower(idx, cache=self._lower_cache)

                out = numpy.min(numpy.broadcast_arrays(
                    left_lower*right_lower, left_lower*right_upper,
                    left_upper*right_lower, left_upper*right_upper,
                ), axis=0).T

            else:
                out = numpy.min([left_lower*right[idx], left_upper*right[idx]], axis=0).T

        else:
            assert isinstance(right, Distribution)
            right_upper = right._get_upper(idx, cache=self._upper_cache)
            right_lower = right._get_lower(idx, cache=self._lower_cache)
            out = numpy.min([left[idx]*right_lower,
                             left[idx]*right_upper], axis=0).T

        return out

    def _upper(self, idx, left, right, cache):
        """
        Distribution upper bounds.

        Example:
            >>> chaospy.Multiply(chaospy.Uniform(-1, 2), -2).upper
            array([2.])
            >>> chaospy.Multiply(chaospy.Uniform(-1, 1), chaospy.Uniform(1, 2)).upper
            array([2.])

        """
        # small hack to deal with sign-flipping boundaries.
        left = self._parameters["left"]
        right = self._parameters["right"]
        if isinstance(left, Distribution):
            left_lower = left._get_lower(idx, cache=self._lower_cache)
            left_upper = left._get_upper(idx, cache=self._upper_cache)

            if isinstance(right, Distribution):
                right_lower = right._get_lower(idx, cache=self._lower_cache)
                right_upper = right._get_upper(idx, cache=self._upper_cache)

                out = numpy.max(numpy.broadcast_arrays(
                    left_lower*right_lower, left_lower*right_upper,
                    left_upper*right_lower, left_upper*right_upper,
                ), axis=0)

            else:
                out = numpy.max([left_lower*right[idx],
                                 left_upper*right[idx]], axis=0)

        else:
            assert isinstance(right, Distribution)
            right_lower = right._get_lower(idx, cache=self._lower_cache)
            right_upper = right._get_upper(idx, cache=self._upper_cache)
            out = numpy.max([left[idx]*right_lower,
                             left[idx]*right_upper], axis=0)

        return out

    def _cdf(self, xloc, idx, left, right, cache):
        if isinstance(left, Distribution):
            left, right = right, left
        left = numpy.broadcast_arrays(left.T, xloc.T)[0].T
        valids = left != 0
        xloc_ = xloc.copy()
        xloc_.T[valids.T] = xloc.T[valids.T]/left.T[valids.T]
        uloc = right._get_fwd(xloc_, idx, cache=cache)
        return numpy.where(left.T >= 0, uloc.T, 1-uloc.T).T

    def _ppf(self, uloc, idx, left, right, cache):
        """
        Point percentile function.

        Example:
            >>> print(chaospy.Uniform().inv([0.1, 0.2, 0.9]))
            [0.1 0.2 0.9]
            >>> print(Multiply(chaospy.Uniform(), 2).inv([0.1, 0.2, 0.9]))
            [0.2 0.4 1.8]
            >>> print(Multiply(2, chaospy.Uniform()).inv([0.1, 0.2, 0.9]))
            [0.2 0.4 1.8]
            >>> dist = chaospy.Multiply([2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> print(dist.inv([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]]))
            [[1.  1.2 1.4]
             [0.5 0.6 0.7]]
            >>> dist = chaospy.Multiply(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> print(dist.inv([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]]))
            [[0.5 0.6 0.7]
             [1.  1.2 1.4]]

        """
        if isinstance(right, Distribution):
            left, right = right, left
        uloc = numpy.where(numpy.asfarray(right).T > 0, uloc.T, 1-uloc.T).T
        xloc = left._get_inv(uloc, idx, cache=cache)
        xloc = (xloc.T*right.T).T

        return xloc

    def _pdf(self, xloc, idx, left, right, cache):
        """
        Probability density function.

        Example:
            >>> print(chaospy.Uniform().pdf([-0.5, 0.5, 1.5, 2.5]))
            [0. 1. 0. 0.]
            >>> print(Multiply(chaospy.Uniform(), 2).pdf([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.5 0.5 0. ]
            >>> print(Multiply(2, chaospy.Uniform()).pdf([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.5 0.5 0. ]
            >>> dist = chaospy.Multiply([2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> print(dist.pdf([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [0.5 0.5 0. ]
            >>> dist = chaospy.Multiply(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> print(dist.pdf([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [0.5 0.5 0. ]

        """
        if isinstance(right, Distribution):
            left, right = right, left
        right = (numpy.asfarray(right).T+numpy.zeros(xloc.shape).T).T
        valids = right != 0
        xloc = xloc.copy()
        xloc.T[valids.T] = xloc.T[valids.T]/right.T[valids.T]
        pdf = left._get_pdf(xloc, idx, cache=cache)
        pdf.T[valids.T] /= right.T[valids.T]
        return numpy.abs(pdf)

    def _mom(self, key, left, right, cache):
        """
        Statistical moments.

        Example:
            >>> print(numpy.around(chaospy.Uniform().mom([0, 1, 2, 3]), 4))
            [1.     0.5    0.3333 0.25  ]
            >>> print(numpy.around(Multiply(chaospy.Uniform(), 2).mom([0, 1, 2, 3]), 4))
            [1.     1.     1.3333 2.    ]
            >>> print(numpy.around(Multiply(2, chaospy.Uniform()).mom([0, 1, 2, 3]), 4))
            [1.     1.     1.3333 2.    ]
            >>> print(numpy.around(Multiply(chaospy.Uniform(), chaospy.Uniform()).mom([0, 1, 2, 3]), 4))
            [1.     0.25   0.1111 0.0625]
        """
        del cache
        if isinstance(left, Distribution):
            if chaospy.shares_dependencies(left, right):
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

    def _ttr(self, kloc, idx, left, right, cache):
        """Three terms recurrence coefficients."""
        del cache
        if isinstance(right, Distribution):
            if isinstance(left, Distribution):
                raise chaospy.StochasticallyDependentError(
                    "product of distributions not feasible: "
                    "{} and {}".format(left, right)
                )
            left, right = right, left
        coeff0, coeff1 = left._get_ttr(kloc, idx)
        return coeff0*right, coeff1*right*right
