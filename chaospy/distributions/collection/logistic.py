"""Generalized logistic type 1 distribution."""
import numpy
from scipy import special, misc

from ..baseclass import SimpleDistribution, ShiftScaleDistribution

class logistic(SimpleDistribution):
    """Generalized logistic type 1 distribution."""

    def __init__(self, c=1):
        super(logistic, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return numpy.e**-x/(1+numpy.e**-x)**(c+1)

    def _cdf(self, x, c):
        return (1+numpy.e**-x)**-c

    def _ppf(self, q, c):
        return -numpy.log(q**(-1./c)-1)


class Logistic(ShiftScaleDistribution):
    """
    Generalized logistic type 1 distribution
    Sech squared distribution

    Args:
        skew (float, Distribution):
            Shape parameter
        shift (float, Distribution):
            Location parameter
        scale (float, Distribution):
            Scale parameter

    Examples:
        >>> distribution = chaospy.Logistic(2, 2, 2)
        >>> distribution
        Logistic(skew=2, scale=2, shift=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([1.5761, 3.0855, 4.4689, 6.2736])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0553, 0.0735, 0.0676, 0.0422])
        >>> distribution.sample(4).round(4)
        array([4.8799, 0.6656, 9.3128, 3.6415])
        >>> distribution.mom(1).round(4)
        4.0
    """
    def __init__(self, skew=1, shift=0, scale=1):
        super(Logistic, self).__init__(
            dist=logistic(skew),
            scale=scale,
            shift=shift,
            repr_args=["skew=%s" % skew],
        )
