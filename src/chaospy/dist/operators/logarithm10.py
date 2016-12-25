"""
Logarithm base 10.
"""
import numpy

from chaospy.dist import Dist

class Log10(Dist):
    """Logarithm base 10."""

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
        return "Log10(%s)" % dist

    def _val(self, graph):
        """Value extraction."""
        if "dist" in graph.keys:
            return numpy.log10(graph.keys["dist"])
        return self

    def _pdf(self, xloc, graph):
        """Probability density function."""
        return graph(10**xloc, graph.dists["dist"])*numpy.log(10)*10**xloc

    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        return graph(10**xloc, graph.dists["dist"])

    def _ppf(self, q, graph):
        """Point percentile function."""
        return numpy.log10(graph(q, graph.dists["dist"]))

    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        lower,upper = graph(10**xloc, graph.dists["dist"])
        return numpy.log10(lower), numpy.log10(upper)


def log10(dist):
    """
    Logarithm base 10.

    Args:
        dist (Dist) : distribution (>=0).
    """
    return Log10(dist)

