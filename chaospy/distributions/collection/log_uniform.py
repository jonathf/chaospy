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
        >>> distribution = chaospy.LogUniform(2, 3, 2, 3)
        >>> distribution
        LogUniform(2, 3, scale=2, shift=3)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([21.05  , 25.0464, 29.9275, 35.8893])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0554, 0.0454, 0.0371, 0.0304])
        >>> distribution.sample(4).round(4)
        array([31.4099, 19.5793, 41.2227, 26.9349])
        >>> distribution.mom(1).round(4)
        28.393
    """

    def __init__(self, lower=0, upper=1, scale=1, shift=0):
        super(LogUniform, self).__init__(
            dist=log_uniform(lower, upper),
            scale=scale,
            shift=shift,
            repr_args=[lower, upper],
        )
