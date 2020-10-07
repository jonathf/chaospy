"""Power log-Normal probability distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class power_log_normal(SimpleDistribution):
    """Power log-Normal probability distribution."""

    def __init__(self, c, s):
        super(power_log_normal, self).__init__(dict(c=c, s=s))

    def _pdf(self, x, c, s):
        norm = (2*numpy.pi)**-.5*numpy.exp(-(numpy.log(x)/s)**2/2.)
        return c/(x*s)*norm*pow(special.ndtr(-numpy.log(x)/s), c*1.-1.)

    def _cdf(self, x, c, s):
        return 1. - pow(special.ndtr(-numpy.log(x)/s), c*1.)

    def _ppf(self, q, c, s):
        return numpy.exp(-s*special.ndtri(pow(1.-q, 1./c)))

    def _lower(self, c, s):
        return 0.


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
        >>> distribution = chaospy.PowerLogNormal(2, 2, 2, 2, 2)
        >>> distribution
        PowerLogNormal(2, mu=2, sigma=2, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([ 3.212 ,  5.2707,  9.5114, 21.2701])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.1347, 0.0711, 0.0317, 0.0092])
        >>> distribution.sample(4).round(4)
        array([11.4445,  2.6512, 69.8654,  6.6177])

    """
    def __init__(self, shape=1, mu=0, sigma=1, scale=1, shift=0):
        dist = ShiftScaleDistribution(
            dist=power_log_normal(shape, sigma), scale=numpy.e**mu)
        super(PowerLogNormal, self).__init__(
            dist=dist,
            scale=scale,
            shift=shift,
            repr_args=[shape, "mu=%s" % mu, "sigma=%s" % sigma]
        )
