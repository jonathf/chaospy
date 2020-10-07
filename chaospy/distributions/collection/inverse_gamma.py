"""Inverse Gamma distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class inverse_gamma(SimpleDistribution):

    def __init__(self, a):
        super(inverse_gamma, self).__init__(dict(a=a))

    def _lower(self, a):
        return 0.

    def _pdf(self, x, a):
        x_ = numpy.where(x, x, 1)
        return numpy.where(x, x_**(-a-1)*numpy.exp(-1./x_)/special.gamma(a), 0)

    def _cdf(self, x, a):
        return numpy.where(x, special.gammaincc(a, 1./numpy.where(x, x, 1)), 0)

    def _ppf(self, q, a):
        return 1./special.gammainccinv(a, q)

    def _mom(self, k, a):
        if k > a:
            return self.upper
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
        >>> distribution = chaospy.InverseGamma(4, scale=2)
        >>> distribution
        InverseGamma(4, scale=2)
        >>> q = numpy.mgrid[0.2:0.8:4j]
        >>> distribution.inv(q).round(4)
        array([0.3626, 0.479 , 0.6228, 0.8708])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([1.7116, 1.6253, 1.147 , 0.5357])
        >>> distribution.sample(4).round(4)
        array([0.673 , 0.3099, 1.4666, 0.5323])
        >>> distribution.mom(1)
        array(6.)

    """

    def __init__(self, shape, scale=1, shift=0):
        super(InverseGamma, self).__init__(
            dist=inverse_gamma(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
