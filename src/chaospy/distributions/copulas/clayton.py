"""Clayton copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class clayton(Archimedean):
    """Clayton copula."""

    def __init__(self, N, theta=1., eps=1e-6):
        Dist.__init__(self, th=float(theta), _length=N, eps=eps)

    def gen(self, x, th):
        return (x**-th-1)/th

    def igen(self, x, th):
        return (1.+th*x)**(-1./th)


class Clayton(Copula):

    def __init__(self, theta=2., eps=1e-6):
        return Copula.__init__(self, dist=dist, trans=clayton(theta, eps))
