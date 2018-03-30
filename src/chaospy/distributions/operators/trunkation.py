"""
Trunkation of distribution.
"""
import numpy

from ..baseclass import Dist


class Trunk(Dist):
    """Trunkation."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        # try:
        #     if isinstance(left, Dist):
        #         assert not left.dependent()
        #         assert numpy.all(left.range()[0] < right)
        #     if isinstance(right, Dist):
        #         assert not right.dependent()
        #         assert numpy.all(left < right.range()[1])
        # except KeyError:
        #     raise ValueError(
        #         "truncation of dependent variables not supported.")

        left_ = not isinstance(left, Dist) or 1 and len(left)
        right_ = not isinstance(right, Dist) or 1 and len(right)
        length = max(left_, right_)

        Dist.__init__(
            self, left=left, right=right, _length=length, _advance=True)

    def _str(self, left, right):
        """String representation."""
        return "(%s<%s)" % (left, right)

    def _pdf(self, xloc, graph):
        """
        Probability density function.

        Example:
            >>> dist = chaospy.trunk(chaospy.Uniform(), 0.6)
            >>> print(dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25]))
            [0.         1.66666667 1.66666667 0.         0.        ]
            >>> dist = chaospy.trunk(chaospy.Uniform(), 0.4)
            >>> print(dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25]))
            [0.  2.5 0.  0.  0. ]
            >>> dist = chaospy.trunk(0.4, chaospy.Uniform())
            >>> print(dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25]))
            [0.         0.         1.66666667 1.66666667 0.        ]
            >>> dist = chaospy.trunk(0.6, chaospy.Uniform())
            >>> print(dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25]))
            [0.  0.  0.  2.5 0. ]
        """
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
            num = (num.T*numpy.ones(xloc.shape[::-1])).T
            out = graph(xloc, dist)/(1-dist.fwd(num))
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
            num = (num.T*numpy.ones(xloc.shape[::-1])).T
            out = graph(xloc, dist)/dist.fwd(num)

        return out

    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        graph_ = graph.copy()
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
            num = (num.T*numpy.ones(xloc.shape[::-1])).T
            uloc1 = graph(xloc, dist)
            uloc2 = graph_(num, dist)
            out = (uloc1-uloc2)/(1-uloc2)
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
            num = (num.T*numpy.ones(xloc.shape[::-1])).T
            uloc1 = graph(xloc, dist)
            uloc2 = graph_(num, dist)
            out = (uloc1)/uloc2

        return out

    def _ppf(self, q, graph):
        """Point percentile function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
            num = (num.T*numpy.ones(q.shape[::-1])).T
            uloc = dist.fwd(num)
            out = graph(q*(1-uloc)+uloc, dist)
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
            num = (num.T*numpy.ones(q.shape[::-1])).T
            uloc = dist.fwd(num)
            out = graph(q*uloc, dist)

        return out

    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
            num = (num.T*numpy.ones(xloc.shape[::-1])).T
            xloc = numpy.where(xloc > num, num, xloc)
            lower, upper = graph(xloc, dist)
            lower = numpy.max([num, lower], 0)
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
            num = (num.T*numpy.ones(xloc.shape[::-1])).T
            xloc = numpy.where(xloc < num, num, xloc)
            lower, upper = graph(xloc, dist)
            upper = numpy.min([num, upper], 0)

        return lower, upper

def trunk(left, right):
    """
    Trunkation.

    Args:
        left (Dist, array_like) : left hand side.
        right (Dist, array_like) : right hand side.
    """
    if not isinstance(left, Dist) and not isinstance(right, Dist):
        return left < right
    return Trunk(left, right)
