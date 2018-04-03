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
    >>> joint2 = chaospy.J(rhs, multiplication)

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
    [ 1.25  -0.5   -0.625]
    >>> print(joint2.mom([(0, 1, 1), (1, 0, 1)]))
    [ 1.25  -2.5   -3.125]
"""
import numpy

from ..baseclass import Dist
from .. import evaluation


class Mul(Dist):
    """Multiplication."""

    def __init__(self, left, right):
        """
        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        # left_ = not isinstance(left, Dist) or 1 and len(left)
        # right_ = not isinstance(right, Dist) or 1 and len(right)
        # length = max(left_, right_)
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
            return left*right, left*right
        else:
            left, right = right, left

        right = numpy.asfarray(right)+numpy.zeros(xloc.shape)
        valids = right != 0
        xloc.T[valids.T] = xloc.T[valids.T]/right.T[valids.T]
        lower, upper = evaluation.evaluate_bound(left, xloc, cache)
        lower, upper = (
            numpy.where(right.T > 0, lower.T, upper.T)*right.T,
            numpy.where(right.T > 0, upper.T, lower.T)*right.T,
        )
        return lower.T, upper.T

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
            return 0.5*(left*right == xloc)
        else:
            left, right = right, left

        right = numpy.asfarray(right)+numpy.zeros(xloc.shape)
        valids = right != 0
        xloc.T[valids.T] = xloc.T[valids.T]/right.T[valids.T]
        output = evaluation.evaluate_forward(left, xloc, cache)
        output.T[~valids.T] = xloc.T[~valids.T] > 0
        output = numpy.where(right.T >= 0, output.T, 1-output.T).T
        return output

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

        uloc = numpy.where(right > 0, uloc, 1-uloc)
        return evaluation.evaluate_inverse(left, uloc, cache)*right

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

        right = numpy.asfarray(right)+numpy.zeros(xloc.shape)
        right = numpy.where(right, right, numpy.inf)
        output = evaluation.evaluate_density(left, xloc/right, cache)
        return output/right

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


class Mvmul(Dist):
    """Multiplication for multivariate variables."""

    def __init__(self, dist, C):
        """
        Constructor.

        Args:
            dist (Dist, array_like) : Probability.
            C (numpy.ndarray) : matrix to multiply with.
        """
        C = C*numpy.eye(len(dist))
        Dist.__init__(self, dist=dist, C=C,
                Ci=numpy.linalg.inv(C),
                _length=len(dist), _advance=True)

    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        return graph(numpy.dot(graph.keys["Ci"], xloc), graph.dists["dist"])

    def _ppf(self, q, graph):
        """Point percentile function."""
        return numpy.dot(graph.keys["C"], graph(q, graph.dists["dist"]))

    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        bnd = graph(xloc, graph.dists["dist"])
        C = graph.keys["C"]
        lower = (numpy.dot(C, bnd[0]).T).T
        upper = (numpy.dot(C, bnd[1]).T).T

        wrong = lower>upper
        out = numpy.where(wrong, upper, lower), numpy.where(wrong, lower, upper)
        return out

    def _val(self, graph):
        """Value extraction."""
        if "dist" in graph.keys:
            return numpy.dot(graph.keys["dist"].T, graph.keys["C"].T).T
        return self

    def _str(self, C, Ci, dist):
        """String representation."""
        return str(dist) + "*" + str(C)

    def _dep(self, graph):
        """Dependency evaluation."""
        dist = graph.dists["dist"]
        S = graph(dist)

        out = [set([]) for _ in range(len(self))]
        C = graph.keys["C"]

        for i in range(len(self)):
            for j in range(len(self)):
                if C[i,j]:
                    out[i].update(S[j])

        return out


def mul(left, right):
    """
    Distribution multiplication.

    Args:
        left (Dist, array_like) : left hand side.
        right (Dist, array_like) : right hand side.
    """
    if left is right:
        return pow(left, 2)

    if isinstance(left, Dist):

        if not isinstance(right, Dist):
            right = numpy.array(right)
            if right.size == 1:
                if right == 1:
                    return left
                if right == 0:
                    return 0.

    elif isinstance(right, Dist):

        left = numpy.array(left)
        if left.size == 1:
            if left == 1:
                return right
            if left == 0:
                return 0.

    else:
        return left*right

    a = not isinstance(left, Dist) or 1 and len(left)
    b = not isinstance(right, Dist) or 1 and len(right)
    length = max(a, b)
    if length == 1:
        return Mul(left, right)
    return Mvmul(dist=left, C=right)
