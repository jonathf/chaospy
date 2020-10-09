"""Fatigue-life distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class fatigue_life(SimpleDistribution):
    """Fatigue-life distribution."""

    def __init__(self, c=0):
        super(fatigue_life, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        output = (x+1)/(2*c*numpy.sqrt(2*numpy.pi*x**3))
        output *= numpy.exp(-(x-1)**2/(2.0*x*c**2))
        output[(x == 0) & numpy.isnan(output)] = 0
        return output

    def _cdf(self, x, c):
        out = special.ndtr(1.0/c*(numpy.sqrt(x)-1.0/numpy.sqrt(x)))
        out = numpy.where(x == 0, 0, out)
        return out

    def _ppf(self, q, c):
        tmp = c*special.ndtri(q)
        out = numpy.where(
            numpy.isfinite(tmp), 0.25*(tmp+numpy.sqrt(tmp**2+4))**2, tmp)
        return out

    def _lower(self, c):
        return (-4*c+numpy.sqrt(16*c**2+1))**2

    def _upper(self, c):
        return (4*c+numpy.sqrt(16*c**2+1))**2


class FatigueLife(ShiftScaleDistribution):
    """
    Fatigue-Life or Birmbaum-Sanders distribution

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.FatigueLife(0.5)
        >>> distribution
        FatigueLife(0.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([ 0.056,  0.659,  0.881,  1.135,  1.519, 17.944])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.869, 0.879, 0.682, 0.377, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.218, 0.553, 2.23 , 0.978])

    """
    def __init__(self, shape=1, scale=1, shift=0):
        super(FatigueLife, self).__init__(
            dist=fatigue_life(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
