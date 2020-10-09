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

    def _lower(self, c):
        return -numpy.log(1e-12**(-1./c)-1)

    def _upper(self, c):
        return -numpy.log((1-1e-12)**(-1./c)-1)


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
        >>> distribution = chaospy.Logistic(15)
        >>> distribution
        Logistic(15)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-1.67 ,  2.178,  2.765,  3.363,  4.201, 30.34 ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.02 , 0.024, 0.02 , 0.012, 0.   ])
        >>> distribution.sample(4).round(3)
        array([3.549, 1.864, 5.682, 2.999])

    """
    def __init__(self, skew=1, shift=0, scale=1):
        super(Logistic, self).__init__(
            dist=logistic(skew),
            scale=scale,
            shift=shift,
            repr_args=[skew],
        )
