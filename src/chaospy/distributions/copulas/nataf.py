"""Nataf (normal) copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class nataf(Dist):
    """Nataf (normal) copula."""

    def __init__(self, R, ordering=None):
        "R symmetric & positive definite matrix"

        if ordering is None:
            ordering = range(len(R))
        ordering = numpy.array(ordering)

        P = numpy.eye(len(R))[ordering]

        R = numpy.dot(P, numpy.dot(R, P.T))
        R = numpy.linalg.cholesky(R)
        R = numpy.dot(P.T, numpy.dot(R, P))
        Ci = numpy.linalg.inv(R)
        Dist.__init__(self, C=R, Ci=Ci, _length=len(R))

    def _cdf(self, x, C, Ci):
        out = special.ndtr(numpy.dot(Ci, special.ndtri(x)))
        return out

    def _ppf(self, q, C, Ci):
        out = special.ndtr(numpy.dot(C, special.ndtri(q)))
        return out

    def _bnd(self, C, Ci):
        return 0.,1.

class Nataf(Copula):

    def __init__(self, R, ordering=None):
        "Nataf (normal) copula"
        return Copula(dist, nataf(R, ordering))
