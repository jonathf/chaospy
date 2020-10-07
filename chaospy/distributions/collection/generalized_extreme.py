"""Generalized extreme value distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class generalized_extreme(SimpleDistribution):
    """Generalized extreme value distribution."""

    def __init__(self, c=1):
        super(generalized_extreme, self).__init__(dict(c=c))

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
    Generalized extreme value distribution
    Fisher-Tippett distribution

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        loc (float, Distribution):
            Location parameter

    Example:
        >>> distribution = chaospy.GeneralizedExtreme(3, 2, 2)
        >>> distribution
        GeneralizedExtreme(3, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([-0.1126,  2.1538,  2.5778,  2.6593])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0386, 0.2382, 1.1497, 8.0333])
        >>> distribution.sample(4).round(4)
        array([ 2.6154, -4.0776,  2.6666,  2.4079])
        >>> distribution.mom(1).round(4)
        -2.9235

    """

    def __init__(self, shape=0, scale=1, shift=0):
        super(GeneralizedExtreme, self).__init__(
            dist=generalized_extreme(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
