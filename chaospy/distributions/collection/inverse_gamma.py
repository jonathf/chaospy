"""Inverse Gamma distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class inverse_gamma(SimpleDistribution):

    def __init__(self, a):
        super(inverse_gamma, self).__init__(dict(a=a))

    def _lower(self, a):
        return 0.

    def _upper(self, a):
        return 1./special.gammainccinv(a, 1-1e-10)

    def _pdf(self, x, a):
        x_ = numpy.where(x, x, 1)
        return numpy.where(x, x_**(-a-1)*numpy.exp(-1./x_)/special.gamma(a), 0)

    def _cdf(self, x, a):
        return numpy.where(x, special.gammaincc(a, 1./numpy.where(x, x, 1)), 0)

    def _ppf(self, q, a):
        return 1./special.gammainccinv(a, q)

    def _mom(self, k, a):
        if k > a:
            return self._upper(a)
        return numpy.prod(a-numpy.arange(1, k.item()+1))


class InverseGamma(ShiftScaleDistribution):
    """
    Inverse-Gamma distribution.

    Args:
        shape (float, Distribution):
            Shape parameter. a>0.
        scale (float, Distribution):
            Scale parameter. scale!=0
        shift (float, Distribution):
            Location of the lower bound.

    Examples:
        >>> distribution = chaospy.InverseGamma(shape=5)
        >>> distribution
        InverseGamma(5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([ 0.   ,  0.149,  0.191,  0.241,  0.324, 38.218])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 4.629, 4.569, 3.352, 1.65 , 0.   ])
        >>> distribution.sample(4).round(3)
        array([0.258, 0.129, 0.508, 0.21 ])
        >>> distribution.mom(1).round(3)
        4.0

    """

    def __init__(self, shape, scale=1, shift=0):
        super(InverseGamma, self).__init__(
            dist=inverse_gamma(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
