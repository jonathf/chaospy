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

Construct joint multiplication distribution::

    >>> lhs = chaospy.Uniform(-1, 0)
    >>> rhs = chaospy.Uniform(-3, -2)
    >>> multiplication = lhs * rhs
    >>> print(multiplication)
    Mul(Uniform(lower=-1, upper=0), Uniform(lower=-3, upper=-2))
    >>> joint1 = chaospy.J(lhs, multiplication)
    >>> print(joint1.lower)
    [-1.  0.]
    >>> print(joint1.upper)
    [0. 3.]
    >>> joint2 = chaospy.J(rhs, multiplication)
    >>> print(joint2.lower)
    [-3.  0.]
    >>> print(joint2.upper)
    [-2.  3.]
    >>> joint3 = chaospy.J(multiplication, lhs)
    >>> print(joint3.lower)
    [ 0. -1.]
    >>> print(joint3.upper)
    [3. 0.]
    >>> joint4 = chaospy.J(multiplication, rhs)
    >>> print(joint4.lower)
    [ 0. -3.]
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

from ..baseclass import Dist
from .. import evaluation


class Mul(Dist):
    """Multiplication."""

    def __init__(self, left, right):
        """
        Args:
            left (Dist, numpy.ndarray) : Left hand side.
            right (Dist, numpy.ndarray) : Right hand side.
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

    def _lower(self, left, right, cache):
        """
        Distribution lower bounds.

        Example:
            >>> print(chaospy.Uniform().lower)
            [0.]
            >>> print(chaospy.Mul(chaospy.Uniform(), 2).lower)
            [0.]
            >>> print(chaospy.Mul(chaospy.Uniform(1, 2), -1).lower)
            [-2.]
            >>> print(chaospy.Mul(2, chaospy.Uniform()).lower)
            [0.]
            >>> print(chaospy.Mul(2, chaospy.Uniform(-1, 0)).lower)
            [-2.]
            >>> print(chaospy.Mul(2, 3).lower)
            [6.]
        """
        if isinstance(left, Dist):
            left_upper = evaluation.evaluate_upper(left, cache=cache)
            left_lower = evaluation.evaluate_lower(left, cache=cache)

            if isinstance(right, Dist):
                right_upper = evaluation.evaluate_upper(right, cache=cache)
                right_lower = evaluation.evaluate_lower(right, cache=cache)

                out = numpy.min(numpy.broadcast_arrays(
                    left_lower*right_lower,
                    left_lower*right_upper,
                    left_upper*right_lower,
                    left_upper*right_upper,
                ), axis=0)

            elif self.matrix:
                out = numpy.min([numpy.dot(left_lower, right),
                                 numpy.dot(left_upper, right)], axis=0)
            else:
                out = numpy.min([left_lower*right, left_upper*right], axis=0)

        elif not isinstance(right, Dist):
            out = left*right

        else:
            right_upper = evaluation.evaluate_upper(right, cache=cache)
            right_lower = evaluation.evaluate_lower(right, cache=cache)
            if self.matrix:
                out = numpy.min([numpy.dot(left, right_lower),
                                 numpy.dot(left, right_upper)], axis=0)
            else:
                out = numpy.min([left*right_lower, left*right_upper], axis=0)

        assert numpy.array(out).shape[:1] in [(), (len(self),)], out
        return out

    def _upper(self, left, right, cache):
        """
        Distribution upper bounds.

        Example:
            >>> print(chaospy.Uniform().upper)
            [1.]
            >>> print(chaospy.Mul(chaospy.Uniform(), 2).upper)
            [2.]
            >>> print(chaospy.Mul(chaospy.Uniform(1, 2), -1).upper)
            [-1.]
            >>> print(chaospy.Mul(2, chaospy.Uniform()).upper)
            [2.]
            >>> print(chaospy.Mul(2, chaospy.Uniform(-1, 0)).upper)
            [0.]
            >>> print(chaospy.Mul(2, 3).upper)
            [6.]
        """
        if isinstance(left, Dist):
            left_lower = evaluation.evaluate_lower(left, cache=cache)
            left_upper = evaluation.evaluate_upper(left, cache=cache)

            if isinstance(right, Dist):
                right_lower = evaluation.evaluate_lower(right, cache=cache)
                right_upper = evaluation.evaluate_upper(right, cache=cache)

                out = numpy.max(numpy.broadcast_arrays(
                    (left_lower.T*right_lower.T).T,
                    (left_lower.T*right_upper.T).T,
                    (left_upper.T*right_lower.T).T,
                    (left_upper.T*right_upper.T).T,
                ), axis=0)

            elif self.matrix:
                out = numpy.max([numpy.dot(left_lower, right),
                                 numpy.dot(left_upper, right)], axis=0)
            else:
                out = numpy.max([left_lower*right, left_upper*right], axis=0)

        elif not isinstance(right, Dist):
            out = left*right

        else:
            right_lower = evaluation.evaluate_lower(right, cache=cache)
            right_upper = evaluation.evaluate_upper(right, cache=cache)
            if self.matrix:
                out = numpy.max([numpy.dot(left, right_lower),
                                 numpy.dot(left, right_upper)], axis=0)
            else:
                out = numpy.max([left*right_lower, left*right_upper], axis=0)

        assert numpy.array(out).shape[:1] in [(), (len(self),)]
        return out


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
        left = evaluation.get_forward_cache(left, cache)
        right = evaluation.get_forward_cache(right, cache)

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
                xloc = xloc.copy()
                xloc.T[valids.T] = xloc.T[valids.T]/left.T[valids.T]

            uloc = evaluation.evaluate_forward(right, xloc, cache=cache)
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
            xloc = xloc.copy()
            xloc.T[valids.T] = xloc.T[valids.T]/right.T[valids.T]

        assert len(left) == len(xloc)
        uloc = evaluation.evaluate_forward(left, xloc, cache=cache)
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
        left = evaluation.get_inverse_cache(left, cache)
        right = evaluation.get_inverse_cache(right, cache)

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
            xloc = evaluation.evaluate_inverse(right, uloc, cache=cache)
            if self.matrix:
                xloc = numpy.dot(left, xloc)
            else:
                xloc *= left
            return xloc

        if not self.matrix:
            uloc = numpy.where(numpy.asfarray(right).T > 0, uloc.T, 1-uloc.T).T
        xloc = evaluation.evaluate_inverse(left, uloc, cache=cache)
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
        left = evaluation.get_forward_cache(left, cache)
        right = evaluation.get_forward_cache(right, cache)

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

            pdf = evaluation.evaluate_density(right, xloc, cache=cache)
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

        pdf = evaluation.evaluate_density(left, xloc, cache=cache)
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
            left = evaluation.evaluate_moment(left, key, cache=cache)
        else:
            left = (numpy.array(left).T**key).T
        if isinstance(right, Dist):
            right = evaluation.evaluate_moment(right, key, cache=cache)
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
            return super(Mul, self).__str__()
        return (self.__class__.__name__ + "(" + str(self.prm["left"]) +
                ", " + str(self.prm["right"]) + ")")

    def __len__(self):
        try:
            return len(self.prm["left"])
        except TypeError:
            return 1

    def _fwd_cache(self, cache):
        left = evaluation.get_forward_cache(self.prm["left"], cache)
        right = evaluation.get_forward_cache(self.prm["right"], cache)
        if not isinstance(left, Dist) and not isinstance(right, Dist):
            return left*right
        return self

    def _inv_cache(self, cache):
        left = evaluation.get_inverse_cache(self.prm["left"], cache)
        right = evaluation.get_inverse_cache(self.prm["right"], cache)
        if not isinstance(left, Dist) and not isinstance(right, Dist):
            return left*right
        return self


def mul(left, right):
    """
    Distribution multiplication.

    Args:
        left (Dist, numpy.ndarray) : left hand side.
        right (Dist, numpy.ndarray) : right hand side.
    """
    from .mv_mul import MvMul
    length = max(left, right)
    if length == 1:
        return Mul(left, right)
    return MvMul(left, right)
