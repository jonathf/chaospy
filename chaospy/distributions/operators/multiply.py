"""
Multiplication of distributions.

Example usage
-------------

Distribution multiplied with a constant::

    >>> distribution = chaospy.Uniform(0, 1)*4
    >>> distribution
    Multiply(Uniform(), 4)
    >>> distribution.sample(5).round(4)
    array([2.6144, 0.46  , 3.8011, 1.9288, 3.4899])
    >>> distribution.fwd([1, 2, 3]).round(4)
    array([0.25, 0.5 , 0.75])
    >>> distribution.inv(distribution.fwd([1, 2, 3])).round(4)
    array([1., 2., 3.])
    >>> distribution.pdf([1, 2, 3]).round(4)
    array([0.25, 0.25, 0.25])
    >>> distribution.mom([1, 2, 3]).round(4)
    array([ 2.    ,  5.3333, 16.    ])
    >>> distribution.ttr([1, 2, 3]).round(4)
    array([[2.    , 2.    , 2.    ],
           [1.3333, 1.0667, 1.0286]])

Construct joint multiplication distribution::

    >>> lhs = chaospy.Uniform(-1, 0)
    >>> rhs = chaospy.Uniform(-3, -2)
    >>> multiplication = lhs * rhs
    >>> multiplication
    Multiply(Uniform(lower=-1, upper=0), Uniform(lower=-3, upper=-2))
    >>> joint1 = chaospy.J(lhs, multiplication)
    >>> joint1.lower
    array([-1., -0.])
    >>> joint1.upper
    array([0., 3.])
    >>> joint2 = chaospy.J(rhs, multiplication)
    >>> joint2.lower
    array([-3., -0.])
    >>> joint2.upper
    array([-2.,  3.])
    >>> joint3 = chaospy.J(multiplication, lhs)
    >>> joint3.lower
    array([-0., -1.])
    >>> joint3.upper
    array([3., 0.])
    >>> joint4 = chaospy.J(multiplication, rhs)
    >>> joint4.lower
    array([-0., -3.])
    >>> joint4.upper
    array([ 3., -2.])

Generate random samples::

    >>> joint1.sample(4).round(4)
    array([[-0.7877, -0.9593, -0.6028, -0.7669],
           [ 2.2383,  2.1172,  1.6532,  1.8345]])
    >>> joint2.sample(4).round(4)
    array([[-2.8177, -2.2565, -2.9304, -2.1147],
           [ 2.6843,  2.1011,  1.2174,  0.0613]])

Forward transformations::

    >>> lcorr = numpy.array([-0.9, -0.5, -0.1])
    >>> rcorr = numpy.array([-2.99, -2.5, -2.01])
    >>> joint1.fwd([lcorr, lcorr*rcorr]).round(4)
    array([[0.1 , 0.5 , 0.9 ],
           [0.99, 0.5 , 0.01]])
    >>> joint2.fwd([rcorr, lcorr*rcorr]).round(4)
    array([[0.01, 0.5 , 0.99],
           [0.9 , 0.5 , 0.1 ]])

Inverse transformations::

    >>> joint1.inv(joint1.fwd([lcorr, lcorr*rcorr])).round(4)
    array([[-0.9  , -0.5  , -0.1  ],
           [ 2.691,  1.25 ,  0.201]])
    >>> joint2.inv(joint2.fwd([rcorr, lcorr*rcorr])).round(4)
    array([[-2.99 , -2.5  , -2.01 ],
           [ 2.691,  1.25 ,  0.201]])
"""
import numpy
import chaospy

from ..baseclass import Distribution, OperatorDistribution


class Multiply(OperatorDistribution):
    """Multiplication."""

    _operator = lambda self, left, right: (left.T*right.T).T

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
            >>> chaospy.Uniform().inv([0.1, 0.2, 0.9])
            array([0.1, 0.2, 0.9])
            >>> Multiply(chaospy.Uniform(), 2).inv([0.1, 0.2, 0.9])
            array([0.2, 0.4, 1.8])
            >>> Multiply(2, chaospy.Uniform()).inv([0.1, 0.2, 0.9])
            array([0.2, 0.4, 1.8])
            >>> dist = chaospy.Multiply(
            ...     [2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> dist.inv([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]])
            array([[1. , 1.2, 1.4],
                   [0.5, 0.6, 0.7]])
            >>> dist = chaospy.Multiply(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> dist.inv([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]])
            array([[0.5, 0.6, 0.7],
                   [1. , 1.2, 1.4]])

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
            >>> chaospy.Uniform().pdf([-0.5, 0.5, 1.5, 2.5])
            array([0., 1., 0., 0.])
            >>> Multiply(chaospy.Uniform(), 2).pdf([-0.5, 0.5, 1.5, 2.5])
            array([0. , 0.5, 0.5, 0. ])
            >>> Multiply(2, chaospy.Uniform()).pdf([-0.5, 0.5, 1.5, 2.5])
            array([0. , 0.5, 0.5, 0. ])
            >>> dist = chaospy.Multiply([2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> dist.pdf([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]])
            array([0.5, 0.5, 0. ])
            >>> dist = chaospy.Multiply(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> dist.pdf([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]])
            array([0.5, 0.5, 0. ])

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
            >>> chaospy.Uniform().mom([0, 1, 2, 3]).round(4)
            array([1.    , 0.5   , 0.3333, 0.25  ])
            >>> Multiply(chaospy.Uniform(), 2).mom([0, 1, 2, 3]).round(4)
            array([1.    , 1.    , 1.3333, 2.    ])
            >>> Multiply(2, chaospy.Uniform()).mom([0, 1, 2, 3]).round(4)
            array([1.    , 1.    , 1.3333, 2.    ])
            >>> Multiply(chaospy.Uniform(), chaospy.Uniform()).mom([0, 1, 2, 3]).round(4)
            array([1.    , 0.25  , 0.1111, 0.0625])

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
