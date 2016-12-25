"""
Logarithm base e.
"""
import numpy

from chaospy.dist import Dist

class Log(Dist):
    """Logarithm base e."""

    def __init__(self, dist):
        """
        Constructor.

        Args:
            dist (Dist) : distribution (>=0).
        """
        assert isinstance(dist, Dist)
        assert numpy.all(dist.range()>=0)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        """String representation."""
        return "Log(%s)" % dist

    def _val(self, graph):
        """Value extraction."""
        if "dist" in graph.keys:
            return numpy.log(graph.keys["dist"])
        return self

    def _pdf(self, xloc, graph):
        """Probability density function."""
        return graph(numpy.e**xloc, graph.dists["dist"])*numpy.e**xloc

    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        return graph(numpy.e**xloc, graph.dists["dist"])

    def _ppf(self, q, graph):
        """Point percentile function."""
        return numpy.log(graph(q, graph.dists["dist"]))

    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        lower,upper = graph(numpy.e**xloc, graph.dists["dist"])
        return numpy.log(lower), numpy.log(upper)


def log(dist):
    """
    Logarithm base e.

    Args:
        dist (Dist) : distribution (>=0).
    """
    return Log(dist)
