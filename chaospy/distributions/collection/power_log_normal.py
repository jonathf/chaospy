"""Power log-Normal probability distribution."""
import numpy
from scipy import special
import chaospy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class power_log_normal(SimpleDistribution):
    """Power log-Normal probability distribution."""

    def __init__(self, c, s):
        super(power_log_normal, self).__init__(dict(c=c, s=s))

    def _pdf(self, x, c, s):
        norm = (2*numpy.pi)**-.5*numpy.exp(-(numpy.log(x)/s)**2/2.)
        out = c/(x*s)*norm*pow(special.ndtr(-numpy.log(x)/s), c*1.-1.)
        out = numpy.where(x == 0, 0, out)
        return out

    def _cdf(self, x, c, s):
        return 1.-pow(special.ndtr(-numpy.log(x)/s), c*1.)

    def _ppf(self, q, c, s):
        return numpy.exp(-s*special.ndtri(pow(1.-q, 1./c)))

    def _lower(self, c, s):
        return 0.

    def _upper(self, c, s):
        return numpy.exp(-s*special.ndtri(pow(1e-12, 1./c)))


class PowerLogNormal(ShiftScaleDistribution):
    """
    Power log-normal distribution

    Args:
        shape (float, Distribution):
            Shape parameter
        mu (float, Distribution):
            Mean in the normal distribution.  Overlaps with scale by
            mu=log(scale)
        sigma (float, Distribution):
            Standard deviation of the normal distribution.
        shift (float, Distribution):
            Location parameter
        scale (float, Distribution):
            Scaling parameter. Overlap with mu in scale=e**mu

    Examples:
        >>> distribution = chaospy.PowerLogNormal(1.5)
        >>> distribution
        PowerLogNormal(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([  0.   ,   0.337,   0.573,   0.898,   1.502, 273.691])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.912, 0.755, 0.488, 0.214, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.017, 0.242, 3.01 , 0.69 ])

    """
    def __init__(self, shape=1, mu=0, sigma=1, scale=1, shift=0):
        dist = ShiftScaleDistribution(
            dist=power_log_normal(shape, sigma), scale=numpy.e**mu)
        super(PowerLogNormal, self).__init__(
            dist=dist,
            scale=scale,
            shift=shift,
            repr_args=[shape]+chaospy.format_repr_kwargs(mu=(mu, 0))+
                              chaospy.format_repr_kwargs(sigma=(sigma, 1)),
        )
