"""Gamma distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class gamma(SimpleDistribution):

    def __init__(self, a=1):
        super(gamma, self).__init__(dict(a=a))

    def _pdf(self, x, a):
        # return x**(a-1)*numpy.e**(-x)/special.gamma(a)
        return numpy.exp(special.xlogy(a-1.0, x) - x - special.gammaln(a))

    def _cdf(self, x, a):
        return special.gammainc(a, x)

    def _ppf(self, q, a):
        return special.gammaincinv(a, q)

    def _mom(self, k, a):
        out = 1.
        for k_ in range(k.item()):
            out *= a+k_
        return out

    def _ttr(self, n, a):
        return 2.*n+a, n*n+n*(a-1)

    def _lower(self, a):
        return 0.

    def _upper(self, a):
        return special.gammaincinv(a, 1-1e-14)


class Gamma(ShiftScaleDistribution):
    """
    Gamma distribution.

    Also an Erlang distribution when shape=k and scale=1./lamb.

    Args:
        shape (float, Distribution):
            Shape parameter. a>0.
        scale (float, Distribution):
            Scale parameter. scale!=0
        shift (float, Distribution):
            Location of the lower bound.

    Examples:
        >>> distribution = chaospy.Gamma(3, scale=0.5)
        >>> distribution
        Gamma(3, scale=0.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([ 0.   ,  0.768,  1.143,  1.553,  2.14 , 19.459])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.508, 0.531, 0.432, 0.254, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.683, 0.587, 3.152, 1.301])
        >>> distribution.mom(1).round(3)
        1.5
        >>> distribution.ttr([0, 1, 2, 3]).round(3)
        array([[1.5 , 2.5 , 3.5 , 4.5 ],
               [0.  , 0.75, 2.  , 3.75]])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Gamma, self).__init__(
            dist=gamma(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )


class Exponential(ShiftScaleDistribution):
    R"""
    Exponential Probability Distribution

    Args:
        scale (float, Distribution):
            Scale parameter. scale!=0
        shift (float, Distribution):
            Location of the lower bound.

    Examples;:
        >>> distribution = chaospy.Exponential()
        >>> distribution
        Exponential()
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([ 0.   ,  0.223,  0.511,  0.916,  1.609, 32.237])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([1. , 0.8, 0.6, 0.4, 0.2, 0. ])
        >>> distribution.sample(4).round(3)
        array([1.06 , 0.122, 3.001, 0.658])
        >>> distribution.mom(1).round(3)
        1.0
        >>> distribution.ttr([1, 2, 3]).round(3)
        array([[3., 5., 7.],
               [1., 4., 9.]])

    """

    def __init__(self, scale=1, shift=0):
        super(Exponential, self).__init__(
            dist=gamma(1),
            scale=scale,
            shift=shift,
            repr_args=[],
        )
