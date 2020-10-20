"""Weibull Distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class weibull(SimpleDistribution):
    """Weibull Distribution."""

    def __init__(self, a=1):
        super(weibull, self).__init__(dict(a=a))

    def _pdf(self, x, a):
        return a*x**(a-1)*numpy.e**(-x**a)

    def _cdf(self, x, a):
        return (1-numpy.e**(-x**a))

    def _ppf(self, q, a):
        return (-numpy.log(1-q+1*(q==1)))**(1./a)*(q!=1)+30.**(1./a)*(q==1)

    def _mom(self, k, a):
        return special.gamma(1.+k*1./a)

    def _lower(self, a):
        return 0.

    def _upper(self, a):
        return 30.**(1./a)


class Weibull(ShiftScaleDistribution):
    """
    Weibull Distribution

    Args:
        shape (float, Distribution):
            Shape parameter.
        scale (float, Distribution):
            Scale parameter.
        shift (float, Distribution):
            Location of lower bound.

    Examples:
        >>> distribution = chaospy.Weibull(2)
        >>> distribution
        Weibull(2)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.472, 0.715, 0.957, 1.269, 5.477])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.756, 0.858, 0.766, 0.507, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.03 , 0.35 , 1.732, 0.811])
        >>> distribution.mom(1).round(4)
        0.8862

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Weibull, self).__init__(
            dist=weibull(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
