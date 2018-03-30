"""
Multiplication of distributions.

Example usage
-------------

Distribution * a constant::

    >>> distribution = chaospy.Uniform(0, 1) * 4
    >>> print(distribution)
    Uniform(lower=0, upper=1)*4
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
    Uniform(lower=-1, upper=0)*Uniform(lower=-3, upper=-2)
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


class Mul(Dist):
    """Multiplication."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        left_ = not isinstance(left, Dist) or 1 and len(left)
        right_ = not isinstance(right, Dist) or 1 and len(right)
        length = max(left_, right_)
        Dist.__init__(
            self, left=left, right=right, _length=length, _advance=True)


    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]

        num = (num + 1.*(num == 0))
        xloc = (xloc.T/num.T).T
        lower_, upper_ = graph(xloc, dist)
        lower = (num.T*(lower_.T*(num.T > 0) + upper_.T*(num.T < 0))).T
        upper = (num.T*(upper_.T*(num.T > 0) + lower_.T*(num.T < 0))).T

        return lower, upper

    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
        xloc = (xloc.T/(num.T + 1.*(num.T == 0))).T
        out = graph(xloc, dist)
        out = (out.T*(num.T > 0) + (1.-out.T)*(num.T < 0) + \
                1.*(xloc.T > num.T)*(num.T == 0)).T
        return out

    def _ppf(self, uloc, graph):
        """Point percentile function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
        uloc = (uloc*(num > 0) + (1.-uloc)*(num <= 0))
        return graph(uloc, dist)*num

    def _pdf(self, xloc, graph):
        """Probability density function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
        num = numpy.where(num, num, numpy.inf)
        return numpy.abs(graph(xloc*1./num, dist)/num)

    def _mom(self, key, graph):
        """Statistical moments."""
        if len(graph.dists) == 2 and\
                graph.dists["left"].dependent(graph.dists["right"]):
            raise NotImplementedError("dependency")

        left = []
        if "left" in graph.dists:
            left.append(graph(key, graph.dists["left"]))
        else:
            left.append((graph.keys["left"].T**key.T).T)
        if "right" in graph.dists:
            left.append(graph(key, graph.dists["right"]))
        else:
            left.append((graph.keys["right"].T**key.T).T)

        return numpy.prod(left[0]*left[1], 0)

    def _ttr(self, order, graph):
        """Three terms recursion coefficients."""
        if "left" in graph.keys and "right" in graph.dists:
            num = graph.keys["left"]
            dist = graph.dists["right"]
        else:
            dist = graph.dists["left"]
            num = graph.keys["right"]

        coeff0, coeff1 = graph(order, dist)
        return coeff0*num, coeff1*num*num

    def _val(self, graph):
        """Value extraction."""
        if len(graph.keys) == 2:
            return graph.keys["left"]*graph.keys["right"]
        return self

    def _str(self, left, right):
        """String representation."""
        return str(left) + "*" + str(right)


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
