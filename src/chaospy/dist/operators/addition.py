"""
Addition.
"""
from scipy.misc import comb
import numpy

from chaospy.dist import Dist
import chaospy.dist.operators


class Add(Dist):
    """Addition."""

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
            self, left=left, right=right, _length=length,
            _advance=True)

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
        lower, upper = graph((xloc.T-num.T).T, dist)
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
        return chaospy.dist.operators.mul(2, left)

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
