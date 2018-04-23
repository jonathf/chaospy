"""Frank copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class frank(Archimedean):
    """Frank copula."""

    def __init__(self, N, theta, eps=1e-6):
        "theta!=0"
        theta = float(theta)
        assert theta!=0
        Dist.__init__(self, th=theta, _length=N, eps=eps)

    def gen(self, x, th):
        return -numpy.log((numpy.e**(-th*x)-1)/(numpy.e**-th-1))
    def igen(self, q, th):
        return -numpy.log(1+numpy.e**-q*(numpy.e**-th-1))/th

class Frank(Copula):

    def __init__(self, theta=1., eps=1e-4):
        return Copula.__init__(self, dist=dist, trans=frank(theta, eps))
