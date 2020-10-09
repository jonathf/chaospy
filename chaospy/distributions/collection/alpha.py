"""Alpha probability distribution."""
import numpy
from scipy import special

import chaospy
from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class alpha(SimpleDistribution):
    """Standard Alpha distribution."""

    def __init__(self, a=1):
        super(alpha, self).__init__(dict(a=a))

    def _cdf(self, x, a):
        return special.ndtr(a-1./x)/special.ndtr(a)

    def _ppf(self, q, a):
        out = 1.0/(a-special.ndtri(q*special.ndtr(a)))
        return numpy.where(q == 1, self._upper(a), out)

    def _pdf(self, x, a):
        return numpy.where(
            x == 0, 0, numpy.e**(-.5*(a-1./x)**2)/
                (numpy.sqrt(2*numpy.pi)*x**2*special.ndtr(a))
        )

    def _lower(self, a):
        return 0.

    def _upper(self, a):
        return 1./(a-special.ndtri((1-1e-10)*special.ndtr(a)))


class Alpha(ShiftScaleDistribution):
    """
    Alpha distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scale Parameter
        shift (float, Distribution):
            Location of lower threshold

    Examples:
        >>> distribution = chaospy.Alpha(6)
        >>> distribution
        Alpha(6)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([ 0.   ,  0.146,  0.16 ,  0.174,  0.194, 63.709])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([ 0.   , 13.104, 15.108, 12.759,  7.449,  0.   ])
        >>> distribution.sample(4).round(3)
        array([0.178, 0.139, 0.23 , 0.165])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Alpha, self).__init__(
            dist=alpha(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
