"""Gamma distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class gamma(DistributionCore):

    def __init__(self, a=1):
        super(gamma, self).__init__(a=a)

    def _pdf(self, x, a):
        return x**(a-1)*numpy.e**(-x)/special.gamma(a)

    def _cdf(self, x, a):
        return special.gammainc(a, x)

    def _ppf(self, q, a):
        return special.gammaincinv(a, q)

    def _mom(self, k, a):
        return special.gamma(a+k)/special.gamma(a)

    def _ttr(self, n, a):
        return 2.*n+a, n*n+n*(a-1)

    def _lower(self, a):
        return 0.


class Gamma(ShiftScale):
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
        >>> distribution = chaospy.Gamma(1, shift=1)
        >>> distribution
        Gamma(1, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([1.2231, 1.5108, 1.9163, 2.6094])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.8, 0.6, 0.4, 0.2])
        >>> distribution.sample(4).round(4)
        array([2.0601, 1.1222, 4.0014, 1.6581])
        >>> distribution.mom(1)
        array(2.)
        >>> distribution.ttr([1, 2, 3]).round(4)
        array([[4., 6., 8.],
               [1., 4., 9.]])
    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Gamma, self).__init__(
            dist=gamma(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )


class Exponential(ShiftScale):
    R"""
    Exponential Probability Distribution

    Args:
        scale (float, Distribution):
            Scale parameter. scale!=0
        shift (float, Distribution):
            Location of the lower bound.

    Examples;:
        >>> distribution = chaospy.Exponential(2, 3)
        >>> distribution
        Exponential(scale=2, shift=3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([3.4463, 4.0217, 4.8326, 6.2189])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.sample(4).round(4)
        array([5.1203, 3.2444, 9.0028, 4.3163])
        >>> distribution.mom(1).round(4)
        5.0
        >>> distribution.ttr([1, 2, 3]).round(4)
        array([[ 9., 13., 17.],
               [ 4., 16., 36.]])
    """

    def __init__(self, scale=1, shift=0):
        super(Exponential, self).__init__(
            dist=gamma(1),
            scale=scale,
            shift=shift,
            repr_args=[],
        )
