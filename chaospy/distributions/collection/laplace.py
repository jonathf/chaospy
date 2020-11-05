"""Laplace Probability Distribution."""
import numpy
from scipy import special
import chaospy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class laplace(SimpleDistribution):
    """Laplace Probability Distribution."""

    def __init__(self):
        super(laplace, self).__init__()

    def _pdf(self, x):
        return numpy.e**-numpy.abs(x)/2

    def _cdf(self, x):
        return (1+numpy.sign(x)*(1-numpy.e**-abs(x)))/2

    def _mom(self, k):
        return special.factorial(k)*((k+1)%2)

    def _ppf(self, x):
        return numpy.where(x > .5, -numpy.log(2*(1-x)), numpy.log(2*x))

    def _lower(self):
        return -20.

    def _upper(self):
        return 20.


class Laplace(ShiftScaleDistribution):
    R"""
    Laplace Probability Distribution

    Args:
        mu (float, Distribution):
            Mean of the distribution.
        scale (float, Distribution):
            Scaling parameter. scale > 0.

    Examples:
        >>> distribution = chaospy.Laplace()
        >>> distribution
        Laplace()
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-20.   ,  -0.916,  -0.223,   0.223,   0.916,  20.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0. , 0.2, 0.4, 0.4, 0.2, 0. ])
        >>> distribution.sample(4).round(3)
        array([ 0.367, -1.47 ,  2.308, -0.036])
        >>> distribution.mom(1).round(3)
        0.0

    """
    def __init__(self, mu=0, sigma=1):
        super(Laplace, self).__init__(
            dist=laplace(),
            scale=sigma,
            shift=mu,
        )
        self._repr_args = (chaospy.format_repr_kwargs(mu=(mu, 0))+
                           chaospy.format_repr_kwargs(sigma=(sigma, 1)))
