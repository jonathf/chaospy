"""Generalized extreme value distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class generalized_extreme(SimpleDistribution):
    """Generalized extreme value distribution."""

    def __init__(self, c=1):
        super(generalized_extreme, self).__init__(dict(c=c))

    def _lower(self, c):
        out = numpy.where(c == 0, -3.5, -numpy.expm1(c*3.5)/c)
        out = numpy.where(c < 0, 1./c, out)
        return out

    def _upper(self, c):
        out = numpy.where(c == 0, 25, -numpy.expm1(-c*25)/c)
        out = numpy.where(c > 0, 1./c, out)
        return out

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


class GeneralizedExtreme(ShiftScaleDistribution):
    """
    Generalized extreme value distribution.

    Also known as the Fisher-Tippett distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        loc (float, Distribution):
            Location parameter

    Example:
        >>> distribution = chaospy.GeneralizedExtreme(0.5)
        >>> distribution
        GeneralizedExtreme(0.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-9.509, -0.537,  0.086,  0.571,  1.055,  2.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.254, 0.383, 0.429, 0.378, 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.696, -0.941,  1.548,  0.292])

    """

    def __init__(self, shape=0, scale=1, shift=0):
        super(GeneralizedExtreme, self).__init__(
            dist=generalized_extreme(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
