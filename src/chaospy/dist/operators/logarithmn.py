"""
Logarithm base N.
"""
import numpy

from chaospy.dist import Dist


class Logn(Dist):
    """Logarithm base N."""

    def __init__(self, dist, n=2):
        """
        Constructor.

        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        assert isinstance(dist, Dist)
        assert numpy.all(dist.range()>=0)
        assert n>0 and n!=1
        Dist.__init__(self, dist=dist, n=n,
                _length=len(dist), _advance=True)

    def _str(self, dist, n):
        """String representation."""
        return "Logn(%s,&s)" % (dist, n)

    def _val(self, graph):
        """Value extraction."""
        if "dist" in graph.keys:
            return numpy.log(graph.keys["dist"])/numpy.log(graph.keys["n"])
        return self

    def _pdf(self, xloc, graph):
        """Probability density function."""
        n = graph.keys["n"]
        return graph(n**xloc, graph.dists["dist"])*numpy.log(n)*n**xloc

    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        return graph(graph.keys["n"]**xloc, graph.dists["dist"])

    def _ppf(self, q, graph):
        """Point percentile function."""
        return numpy.log(graph(q, graph.dists["dist"]))/graph.keys["n"]

    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        n = graph.keys["n"]
        lower,upper = graph(n**xloc, graph.dists["dist"])
        return numpy.log(lower)/numpy.log(n),\
                numpy.log(upper)/numpy.log(n)

def logn(dist, order):
    """
    Logarithm base N.

    Args:
        dist (Dist) : distribution (>=0)
        order (int) : logarithm base.
    """
    return Logn(dist, order)
