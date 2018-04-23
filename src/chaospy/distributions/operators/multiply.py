"""
Multiplication of distributions.

Example usage
-------------

Distribution * a constant::

    >>> distribution = chaospy.Uniform(0, 1) * 4
    >>> print(distribution)
    Mul(Uniform(lower=0, upper=1), 4)
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

Construct joint addition distribution::

    >>> lhs = chaospy.Uniform(-1, 0)
    >>> rhs = chaospy.Uniform(-3, -2)
    >>> multiplication = lhs * rhs
    >>> print(multiplication)
    Mul(Uniform(lower=-1, upper=0), Uniform(lower=-3, upper=-2))
    >>> joint1 = chaospy.J(lhs, multiplication)
    >>> print(joint1.range())
    [[-1.  0.]
     [ 0.  3.]]
    >>> joint2 = chaospy.J(rhs, multiplication)
    >>> print(joint2.range())
    [[-3. -0.]
     [-2.  3.]]
    >>> joint3 = chaospy.J(multiplication, lhs)
    >>> print(joint3.range())
    [[ 0. -1.]
     [ 3.  0.]]
    >>> joint4 = chaospy.J(multiplication, rhs)
    >>> print(joint4.range())
    [[-0. -3.]
     [ 3. -2.]]

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

Raw moments::

    >>> print(joint1.mom([(0, 1, 1), (1, 0, 1)]))
    [ 1.5  -0.5  -0.75]
    >>> print(joint2.mom([(0, 1, 1), (1, 0, 1)]))
    [ 1.5  -2.5  -3.75]
    >>> print(joint3.mom([(0, 1, 1), (1, 0, 1)]))
    [-0.5   1.5  -0.75]
    >>> print(joint4.mom([(0, 1, 1), (1, 0, 1)]))
    [-2.5   1.5  -3.75]
"""
import numpy

from ..baseclass import Dist
from .. import evaluation, deprecations


class Mul(Dist):
    """Multiplication."""

    def __init__(self, left, right):
        """
        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        self.matrix = False
        if (isinstance(left, Dist) and
                len(left) > 1 and
                not isinstance(right, Dist)):
            right = right*numpy.eye(len(left))
            self.matrix = True

        elif (isinstance(right, Dist) and
                len(right) > 1 and
                not isinstance(left, Dist)):
            left = left*numpy.eye(len(right))
            self.matrix = True

        Dist.__init__(self, left=left, right=right)


    def _bnd(self, xloc, left, right, cache):
        """
        Distribution bounds.

        Example:
            >>> print(chaospy.Uniform().range([-2, 0, 2, 4]))
            [[0. 0. 0. 0.]
             [1. 1. 1. 1.]]
            >>> print(Mul(chaospy.Uniform(), 2).range([-2, 0, 2, 4]))
            [[0. 0. 0. 0.]
             [2. 2. 2. 2.]]
            >>> print(Mul(2, chaospy.Uniform()).range([-2, 0, 2, 4]))
            [[0. 0. 0. 0.]
             [2. 2. 2. 2.]]
            >>> print(Mul(2, 2).range([-2, 0, 2, 4]))
            [[4. 4. 4. 4.]
             [4. 4. 4. 4.]]
            >>> dist = chaospy.Mul(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> print(dist.range([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [[[0. 0. 0.]
              [0. 0. 0.]]
            <BLANKLINE>
             [[1. 1. 1.]
              [2. 2. 2.]]]
            >>> dist = chaospy.Mul([2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> print(dist.range([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [[[0. 0. 0.]
              [0. 0. 0.]]
            <BLANKLINE>
             [[2. 2. 2.]
              [1. 1. 1.]]]
            >>> dist = chaospy.Mul(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> print(dist.range([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [[[0. 0. 0.]
              [0. 0. 0.]]
            <BLANKLINE>
             [[1. 1. 1.]
              [2. 2. 2.]]]
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
            return numpy.dot(left, right), numpy.dot(left, right)
        else:
            left = numpy.asfarray(left)
            if self.matrix:
                Ci = numpy.linalg.inv(left)
                xloc = numpy.dot(Ci, xloc)
                assert len(xloc) == len(right)

            elif len(left.shape) == 3:
                left_ = numpy.mean(left, 0)
                valids = left_ != 0
                xloc.T[valids.T] = xloc.T[valids.T]/left_.T[valids.T]

            else:
                left = (left.T+numpy.zeros(xloc.shape).T).T
                valids = left != 0
                xloc.T[valids.T] = xloc.T[valids.T]/left.T[valids.T]

            assert len(xloc) == len(right)
            lower, upper = evaluation.evaluate_bound(right, xloc, cache)
            if self.matrix:
                lower = numpy.dot(lower.T, left.T).T
                upper = numpy.dot(upper.T, left.T).T

            elif len(left.shape) == 3:
                lower = numpy.where(left[0]*lower > 0, left[0]*lower, left[1]*lower)
                upper = numpy.where(left[1]*upper > 0, left[1]*upper, left[0]*upper)
                lower, upper = (
                    numpy.where(lower < upper, lower, upper),
                    numpy.where(lower < upper, upper, lower),
                )
                lower[(left[0] < 0) & (lower > 0)] = 0.
                assert len(lower) == len(right)

            else:
                lower *= left
                upper *= left
            lower, upper = (
                numpy.where(lower < upper, lower, upper),
                numpy.where(lower < upper, upper, lower),
            )
            return lower, upper


        right = numpy.asfarray(right)
        if self.matrix:
            Ci = numpy.linalg.inv(right)
            xloc = numpy.dot(xloc.T, Ci.T).T
            assert len(left) == len(xloc)

        elif len(right.shape) == 3:
            right_ = numpy.mean(right, 0)
            valids = right_ != 0
            xloc.T[valids.T] = xloc.T[valids.T]/right_.T[valids.T]

        else:
            right = (right.T+numpy.zeros(xloc.shape).T).T
            valids = right != 0
            xloc.T[valids.T] = xloc.T[valids.T]/right.T[valids.T]

        assert len(left) == len(xloc)
        lower, upper = evaluation.evaluate_bound(left, xloc, cache)
        if self.matrix:
            lower = numpy.dot(lower.T, right.T).T
            upper = numpy.dot(upper.T, right.T).T

        elif len(right.shape) == 3:
            lower = numpy.where(right[0]*lower > 0, right[0]*lower, right[1]*lower)
            upper = numpy.where(right[1]*upper > 0, right[1]*upper, right[0]*upper)

            lower, upper = (
                numpy.where(lower < upper, lower, upper),
                numpy.where(lower < upper, upper, lower),
            )
            lower[(right[0] < 0) & (lower > 0)] = 0.

        else:
            lower *= right
            upper *= right
        lower, upper = (
            numpy.where(lower < upper, lower, upper),
            numpy.where(lower < upper, upper, lower),
        )
        assert lower.shape == xloc.shape
        assert upper.shape == xloc.shape
        return lower, upper

    def _cdf(self, xloc, left, right, cache):
        """
        Cumulative distribution function.

        Example:
            >>> print(chaospy.Uniform().fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.5 1.  1. ]
            >>> print(Mul(chaospy.Uniform(), 2).fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.   0.25 0.75 1.  ]
            >>> print(Mul(2, chaospy.Uniform()).fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.   0.25 0.75 1.  ]
            >>> print(Mul(1, 1.5).fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.  0.5 1. ]
            >>> dist = chaospy.Mul([2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> print(dist.fwd([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [[0.25 0.3  0.75]
             [0.5  0.6  1.  ]]
            >>> dist = chaospy.Mul(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> print(dist.fwd([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [[0.5  0.6  1.  ]
             [0.25 0.3  0.75]]
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
            if self.matrix:
                return 0.5*(numpy.dot(left, right) == xloc)
            return 0.5*(left*right == xloc)
        else:
            if self.matrix:
                Ci = numpy.linalg.inv(left)
                xloc = numpy.dot(Ci, xloc)
                assert len(xloc) == len(numpy.dot(Ci, xloc))

            else:
                left = (numpy.asfarray(left).T+numpy.zeros(xloc.shape).T).T
                valids = left != 0
                xloc.T[valids.T] = xloc.T[valids.T]/left.T[valids.T]

            uloc = evaluation.evaluate_forward(right, xloc, cache)
            if not self.matrix:
                uloc = numpy.where(left.T >= 0, uloc.T, 1-uloc.T).T
            assert uloc.shape == xloc.shape
            return uloc

        if self.matrix:
            Ci = numpy.linalg.inv(right)
            xloc = numpy.dot(xloc.T, Ci).T
        else:
            right = (numpy.asfarray(right).T+numpy.zeros(xloc.shape).T).T
            valids = right != 0
            xloc.T[valids.T] = xloc.T[valids.T]/right.T[valids.T]

        assert len(left) == len(xloc)
        uloc = evaluation.evaluate_forward(left, xloc, cache)
        if not self.matrix:
            uloc = numpy.where(right.T >= 0, uloc.T, 1-uloc.T).T
        assert uloc.shape == xloc.shape
        return uloc

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
            >>> print(Mul(2, 2).inv([0.1, 0.2, 0.9]))
            [4. 4. 4.]
            >>> dist = chaospy.Mul([2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> print(dist.inv([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]]))
            [[1.  1.2 1.4]
             [0.5 0.6 0.7]]
            >>> dist = chaospy.Mul(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> print(dist.inv([[0.5, 0.6, 0.7], [0.5, 0.6, 0.7]]))
            [[0.5 0.6 0.7]
             [1.  1.2 1.4]]
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
            if self.matrix:
                return numpy.dot(left, right)
            return left*right

        else:
            if not self.matrix:
                uloc = numpy.where(numpy.asfarray(left).T > 0, uloc.T, 1-uloc.T).T
            xloc = evaluation.evaluate_inverse(right, uloc, cache)
            if self.matrix:
                xloc = numpy.dot(left, xloc)
            else:
                xloc *= left
            return xloc

        if not self.matrix:
            uloc = numpy.where(numpy.asfarray(right).T > 0, uloc.T, 1-uloc.T).T
        xloc = evaluation.evaluate_inverse(left, uloc, cache)
        if self.matrix:
            xloc = numpy.dot(xloc.T, right).T
        else:
            xloc *= right
        assert uloc.shape == xloc.shape
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
            >>> print(Mul(1, 1.5).pdf([-0.5, 0.5, 1.5, 2.5])) # Dirac logic
            [ 0.  0. inf  0.]
            >>> dist = chaospy.Mul([2, 1], chaospy.Iid(chaospy.Uniform(), 2))
            >>> print(dist.pdf([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [0.5 0.5 0. ]
            >>> dist = chaospy.Mul(chaospy.Iid(chaospy.Uniform(), 2), [1, 2])
            >>> print(dist.pdf([[0.5, 0.6, 1.5], [0.5, 0.6, 1.5]]))
            [0.5 0.5 0. ]
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
            if self.matrix:
                Ci = numpy.linalg.inv(left)
                xloc = numpy.dot(Ci, xloc)

            else:
                left = (numpy.asfarray(left).T+numpy.zeros(xloc.shape).T).T
                valids = left != 0
                xloc.T[valids.T] = xloc.T[valids.T]/left.T[valids.T]

            pdf = evaluation.evaluate_density(right, xloc, cache)
            if self.matrix:
                pdf = numpy.dot(Ci, pdf)
            else:
                pdf.T[valids.T] /= left.T[valids.T]
            return pdf

        if self.matrix:
            Ci = numpy.linalg.inv(right)
            xloc = numpy.dot(xloc.T, Ci).T
        else:
            right = (numpy.asfarray(right).T+numpy.zeros(xloc.shape).T).T
            valids = right != 0
            xloc.T[valids.T] = xloc.T[valids.T]/right.T[valids.T]
            xloc.T[~valids.T] = numpy.inf

        pdf = evaluation.evaluate_density(left, xloc, cache)
        if self.matrix:
            pdf = numpy.dot(pdf.T, Ci).T
        else:
            pdf.T[valids.T] /= right.T[valids.T]
        assert pdf.shape == xloc.shape
        return pdf

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
            >>> print(numpy.around(Mul(2, 2).mom([0, 1, 2, 3]), 4))
            [ 1.  4. 16. 64.]
        """
        if evaluation.get_dependencies(left, right):
            raise evaluation.DependencyError(
                "sum of dependent distributions not feasible: "
                "{} and {}".format(left, right)
            )

        if isinstance(left, Dist):
            left = evaluation.evaluate_moment(left, key, cache)
        else:
            left = (numpy.array(left).T**key).T
        if isinstance(right, Dist):
            right = evaluation.evaluate_moment(right, key, cache)
        else:
            right = (numpy.array(right).T**key).T
        return numpy.sum(left*right)

    def _ttr(self, kloc, left, right, cache):
        """Three terms recursion coefficients."""
        if isinstance(left, Dist) and isinstance(right, Dist):
            raise evaluation.DependencyError(
                "product of distributions not feasible: "
                "{} and {}".format(left, right)
            )
        elif isinstance(right, Dist):
            left, right = right, left

        coeff0, coeff1 = evaluation.evaluate_recurrence_coefficients(
            left, kloc, cache=cache)
        right = numpy.asarray(right)
        return coeff0*right, coeff1*right*right

    def __str__(self):
        if self._repr is not None:
            return super().__str__()
        return (self.__class__.__name__ + "(" + str(self.prm["left"]) +
                ", " + str(self.prm["right"]) + ")")

    def __len__(self):
        try:
            return len(self.prm["left"])
        except TypeError:
            return 1



@deprecations.deprecation_warning
def mul(left, right):
    """
    Distribution multiplication.

    Args:
        left (Dist, array_like) : left hand side.
        right (Dist, array_like) : right hand side.
    """
    from .mv_mul import MvMul
    length = max(left, right)
    if length == 1:
        return Mul(left, right)
    return MvMul(left, right)
