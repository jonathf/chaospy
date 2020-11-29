"""Wald distribution."""
import numpy
from scipy import special
import chaospy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class wald(SimpleDistribution):
    """Wald distribution."""

    def __init__(self, mu):
        super(wald, self).__init__(dict(mu=mu))

    def _pdf(self, x, mu):
        out = numpy.zeros(x.shape)
        indices = x > 0
        out[indices] = 1.0/numpy.sqrt(2*numpy.pi*x[indices])
        out[indices] *= numpy.exp(-(1-mu*x[indices])**2.0 / (2*x[indices]*mu**2.0))
        return out

    def _cdf(self, x, mu):
        trm1 = 1./mu-x
        trm2 = 1./mu+x
        isqx = numpy.tile(numpy.inf, x.shape)
        indices = x > 0
        isqx[indices] = 1./numpy.sqrt(x[indices])
        out = 1.-special.ndtr(isqx*trm1)
        out -= numpy.exp(2.0/mu)*special.ndtr(-isqx*trm2)
        out = numpy.where(x == numpy.inf, 1, out)
        out = numpy.where(x == -numpy.inf, 0, out)
        return out

    def _lower(self, mu):
        return 0.

    def _upper(self, mu):
        xloc = numpy.full_like(mu, 5.)
        indices = 1-self._cdf(xloc, mu) > 1e-15
        while numpy.any(indices):
            idx1 = 1-self._cdf(xloc+indices*5., mu) > 1e-15
            idx2 = 1-self._cdf(xloc+indices*25., mu) > 1e-15
            xloc[indices] += numpy.where(idx2[indices], 25., 5.)
            indices[idx1 & ~idx2] = False
        return xloc


class Wald(ShiftScaleDistribution):
    """
    Wald distribution.

    Reciprocal inverse Gaussian distribution.

    Args:
        mu (float, Distribution):
            Mean of the normal distribution
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Wald(0.5)
        >>> distribution
        Wald(0.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([ 0.   ,  1.416,  2.099,  2.94 ,  4.287, 60.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.297, 0.275, 0.2  , 0.105, 0.   ])
        >>> distribution.sample(4).round(3)
        array([3.246, 3.984, 2.721, 5.34 ])

    """

    def __init__(self, mu=1, scale=1, shift=0):
        super(Wald, self).__init__(
            dist=wald(mu),
            scale=scale,
            shift=shift,
            repr_args=[mu],
        )
