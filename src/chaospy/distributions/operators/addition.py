"""
Addition operator.

Example usage
-------------

Distribution and a constant::

    >>> distribution = chaospy.Normal(0, 1) + 10
    >>> print(distribution)
    (Normal(mu=0, sigma=1)+10)
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
    (Uniform(lower=2, upper=3)+Uniform(lower=3, upper=4))
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
from scipy.misc import comb
import numpy

from ..baseclass import Dist
from .multiply import mul


class Add(Dist):
    """Addition."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        try:
            left_ = len(left)
        except TypeError:
            left_ = 1
        except IndexError:
            left_ = 1
        try:
            right_ = len(right)
        except TypeError:
            right_ = 1
        except IndexError:
            right_ = 1
        length = max(left_, right_)
        Dist.__init__(self, left=left, right=right, _length=length, _advance=True)

    def _str(self, left, right):
        """String representation."""
        return "(%s+%s)" % (left, right)

    def _val(self, graph):
        """Value extraction."""
        if len(graph.keys) == 2:
            return graph.keys["left"] + graph.keys["right"]
        return self

    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
        lower, upper = graph((xloc.T).T, dist)
        return (lower.T+num.T).T, (upper.T+num.T).T

    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
        return graph((xloc.T-num.T).T, dist)

    def _pdf(self, xloc, graph):
        """Probability density function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
        return graph((xloc.T-num.T).T, dist)

    def _ppf(self, uloc, graph):
        """Point percentile function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
        return (graph(uloc, dist).T + num.T).T

    def _mom(self, keys, graph):
        """Statistical moments."""
        if len(graph.dists) == 2 and \
                graph.dists["left"].dependent(graph.dists["right"]):
            raise NotImplementedError("dependency")

        dim = len(self)
        kmax = numpy.max(keys, 1)+1
        keys_ = numpy.mgrid[[slice(0, _, 1) for _ in kmax]]
        keys_ = keys_.reshape(dim, int(keys_.size/dim))

        left = []
        if "left" in graph.dists:
            left.append(graph(keys_, graph.dists["left"]))
        else:
            left.append(graph.keys["left"]**keys_.T)
        if "right" in graph.dists:
            left.append(graph(keys_, graph.dists["right"]))
        else:
            left.append(graph.keys["right"]**keys_.T)

        if dim == 1:
            left[0] = left[0].flatten()
            left[1] = left[1].flatten()
            keys = keys.flatten()

        out = 0.
        for idx in range(keys_.shape[1]):
            key = keys_.T[idx]
            coef = comb(keys.T, key)
            out = out + coef*left[0][idx]*left[1][keys-idx]*(key <= keys.T)

        if dim > 1:
            out = numpy.prod(out, 1)
        return out

    def _ttr(self, order, graph):
        """Three terms recursion coefficients."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]

        coeffs0, coeffs1 = graph(order, dist)
        return coeffs0+num, coeffs1


def add(left, right):
    """
    Distribution addition.

    Args:
        left (Dist, array_like) : left hand side.
        right (Dist, array_like) : right hand side.
    """
    if left is right:
        return mul(2, left)

    if isinstance(left, Dist):
        if not isinstance(right, Dist):
            right = numpy.array(right)
            if right.size == 1 and right == 0:
                return left

    elif isinstance(right, Dist):
        left = numpy.array(left)
        if left.size == 1 and left == 0:
            return right

    else:
        return left+right

    return Add(left=left, right=right)
