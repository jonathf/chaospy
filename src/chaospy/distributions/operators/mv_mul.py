"""Multiplication for multivariate variables."""
import numpy

from ..baseclass import Dist
from .. import evaluation, deprecations


class MvMul(Dist):
    """Multiplication for multivariate variables."""

    def __init__(self, dist, C):
        """
        Constructor.

        Args:
            dist (Dist, array_like) : Probability.
            C (numpy.ndarray) : matrix to multiply with.
        """
        if isinstance(dist, Dist):
            if isinstance(C, Dist):
                raise evaluation.DependencyError(
                    "multiplication of two multivariate "
                    "distributions is not supported"
                )
        C = C*numpy.eye(len(dist))
        Ci = numpy.linalg.inv(C)
        Dist.__init__(self, dist=dist, C=C, Ci=Ci)

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


@deprecations.deprecation_warning
def Mvmul(left, right):
    return MvMul(left, right)
