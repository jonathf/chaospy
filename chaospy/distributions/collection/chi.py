"""Chi distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class chi(SimpleDistribution):
    """Chi distribution."""

    def __init__(self, df=1):
        super(chi, self).__init__(dict(df=df))

    def _pdf(self, x, df):
        return x**(df-1)*numpy.exp(-x*x*.5)/2**(df*.5-1)/special.gamma(df*.5)

    def _cdf(self, x, df):
        return special.gammainc(df*0.5,0.5*x*x)

    def _ppf(self, q, df):
        return numpy.sqrt(2*special.gammaincinv(df*0.5, q))

    def _lower(self, df):
        return numpy.sqrt(2*special.gammaincinv(df*0.5, 1e-12))

    def _upper(self, df):
        return numpy.sqrt(2*special.gammaincinv(df*0.5, 1-1e-12))

    def _mom(self, k, df):
        return 2**(.5*k)*special.gamma(.5*(df+k))/special.gamma(.5*df)


class Chi(ShiftScaleDistribution):
    """
    Chi distribution.

    Args:
        df (float, Distribution):
            Degrees of freedom
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Chi(1.5)
        >>> distribution
        Chi(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.472, 0.791, 1.127, 1.568, 7.294])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.596, 0.631, 0.546, 0.355, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.229, 0.321, 2.234, 0.924])
        >>> distribution.mom(1).round(3)
        1.046

    """

    def __init__(self, df=1, scale=1, shift=0):
        super(Chi, self).__init__(
            dist=chi(df),
            scale=scale,
            shift=shift,
            repr_args=[df],
        )


class Maxwell(ShiftScaleDistribution):
    """
    Maxwell-Boltzmann distribution
    Chi distribution with 3 degrees of freedom

    Args:
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Maxwell()
        >>> distribution
        Maxwell()
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 1.003, 1.367, 1.716, 2.154, 7.676])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.485, 0.586, 0.539, 0.364, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.819, 0.806, 2.798, 1.507])
        >>> distribution.mom(1).round(3)
        1.596

    """

    def __init__(self, scale=1, shift=0):
        super(Maxwell, self).__init__(
            dist=chi(3),
            scale=scale,
            shift=shift,
            repr_args=[],
        )


class Rayleigh(ShiftScaleDistribution):
    """
    Rayleigh distribution

    Args:
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Rayleigh()
        >>> distribution
        Rayleigh()
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.668, 1.011, 1.354, 1.794, 7.434])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.534, 0.606, 0.541, 0.359, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.456, 0.494, 2.45 , 1.147])
        >>> distribution.mom(1).round(3)
        1.253

    """
    def __init__(self, scale=1, shift=0):
        super(Rayleigh, self).__init__(
            dist=chi(2),
            scale=scale,
            shift=shift,
            repr_args=[],
        )
