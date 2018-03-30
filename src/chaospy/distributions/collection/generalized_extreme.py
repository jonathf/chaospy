"""Generalized extreme value distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class generalized_extreme(Dist):
    """Generalized extreme value distribution."""

    def __init__(self, c=1):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        cx = c*x
        logex2 = numpy.where(c == 0, 0., numpy.log1p(-cx))
        logpex2 = numpy.where(c == 0, -x, logex2/c)
        pex2 = numpy.exp(logpex2)
        logpdf = numpy.where((cx==1) | (cx==-numpy.inf), -numpy.inf, -pex2+logpex2-logex2)
        numpy.putmask(logpdf,(c==1) & (x==1),0.0)
        return numpy.exp(logpdf)

    def _cdf(self, x, c):
        loglogcdf = numpy.where(c == 0, -x, numpy.log1p(-c*x)/c)
        return numpy.exp(-numpy.exp(loglogcdf))

    def _ppf(self, q, c):
        x = -numpy.log(-numpy.log(q))
        return numpy.where(c == 0, x, -numpy.expm1(-c*x)/c)

    def _bnd(self, c):
        return self._ppf(1e-8, c), self._ppf(1-1e-8, c)


class GeneralizedExtreme(Add):
    """
    Generalized extreme value distribution
    Fisher-Tippett distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter

    Example:
        >>> distribution = chaospy.GeneralizedExtreme(3, 2, 2)
        >>> print(distribution)
        GeneralizedExtreme(loc=2, scale=2, shape=3)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [-0.1126  2.1538  2.5778  2.6593]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.0386 0.2382 1.1497 8.0333]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 2.6154 -4.0776  2.6666  2.4079]
    """
    def __init__(self, shape=0, scale=1, loc=0):
        self._repr = {"shape": shape, "scale": scale, "loc": loc}
        Add.__init__(self, left=generalized_extreme(shape)*scale, right=loc)
