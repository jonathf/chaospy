"""T-Copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class t_copula(Dist):
    """T-Copula."""

    def __init__(self, a, R):
        from ..cores import mvstudentt, student_t
        self.MV = mvstudentt(a, numpy.zeros(len(R)), R)
        self.UV = student_t(a)
        Dist.__init__(self, _length=len(R))

    def _cdf(self, x):
        out = self.MV.fwd(self.UV.inv(x))
        return out

    def _ppf(self, q):
        out = self.MV.inv(q)
        out = self.UV.fwd(out)
        return out

    def _bnd(self):
        return 0.,1.

class TCopula(Copula):

    def __init__(self, a, R):
        return Copula(dist, t_copula(a, R))
