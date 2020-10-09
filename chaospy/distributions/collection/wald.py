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
        return chaospy.approximate_inverse(
            distribution=self,
            idx=0,
            qloc=numpy.array([1-1e-15]),
            parameters=dict(mu=mu),
            bounds=(0., 100*(numpy.exp(mu)+numpy.exp(1/mu))),
        )


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
        >>> distribution = chaospy.Wald(2, 2, 2)
        >>> distribution
        Wald(2, scale=2, shift=2)
        >>> distribution.upper.round(4)
        array([905.7777])
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.7154, 3.45  , 4.5777, 6.6902])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.3242, 0.2262, 0.138 , 0.063 ])
        >>> distribution.sample(4).round(4)
        array([3.6277, 4.5902, 3.087 , 5.2544])
    """

    def __init__(self, mu=1, scale=1, shift=0):
        super(Wald, self).__init__(
            dist=wald(mu),
            scale=scale,
            shift=shift,
            repr_args=[mu],
        )
