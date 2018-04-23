"""Joe copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class joe(Archimedean):
    """Joe copula."""

    def __init__(self, N, theta, eps=1e-6):
        "theta in [1,inf)"
        theta = float(theta)
        assert theta>=1
        Dist.__init__(self, th=theta, _length=N, eps=eps)

    def gen(self, x, th):
        return -numpy.log(1-(1-x)**th)

    def igen(self, q, th):
        return 1-(1-numpy.e**-q)**(1/th)

class Joe(Copula):

    def __init__(self, theta=2., eps=1e-6):
        "Joe copula"
        return Copula(dist, joe(len(dist), theta, eps))
