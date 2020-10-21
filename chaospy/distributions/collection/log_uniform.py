"""Log-uniform distribution."""
import numpy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class log_uniform(SimpleDistribution):
    """Log-uniform distribution."""

    def __init__(self, lo=0, up=1):
        super(log_uniform, self).__init__(dict(lo=lo, up=up))

    def _pdf(self, x, lo, up):
        return 1./(x*(up-lo))

    def _cdf(self, x, lo, up):
        return (numpy.log(x)-lo)/(up-lo)

    def _ppf(self, q, lo, up):
        return numpy.e**(q*(up-lo) + lo)

    def _lower(self, lo, up):
        return numpy.e**lo

    def _upper(self, lo, up):
        return numpy.e**up

    def _mom(self, k, lo, up):
        return ((numpy.e**(up*k)-numpy.e**(lo*k))/((up-lo)*(k+(k==0))))**(k!=0)


class LogUniform(ShiftScaleDistribution):
    """
    Log-uniform distribution

    Args:
        lower (float, Distribution):
            Location of lower threshold of uniform distribution.
        upper (float, Distribution):
            Location of upper threshold of uniform distribution.
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.LogUniform(0.5, 1.5)
        >>> distribution
        LogUniform(0.5, 1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([1.649, 2.014, 2.46 , 3.004, 3.669, 4.482])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.607, 0.497, 0.407, 0.333, 0.273, 0.223])
        >>> distribution.sample(4).round(3)
        array([3.17 , 1.85 , 4.264, 2.67 ])
        >>> distribution.mom(1).round(3)
        2.833

    """

    def __init__(self, lower=0, upper=1, scale=1, shift=0):
        super(LogUniform, self).__init__(
            dist=log_uniform(lower, upper),
            scale=scale,
            shift=shift,
            repr_args=[lower, upper],
        )
