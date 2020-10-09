"""Mielke's beta-kappa distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class mielke(SimpleDistribution):
    """Mielke's beta-kappa distribution."""

    def __init__(self, k, s):
        super(mielke, self).__init__(dict(k=k, s=s))

    def _pdf(self, x, k, s):
        return k*x**(k-1.)/(1.+x**s)**(1.+k*1./s)

    def _cdf(self, x, k, s):
        return x**k/(1.+x**s)**(k*1./s)

    def _ppf(self, q, k, s):
        qsk = pow(q, s*1./k)
        return pow(qsk/(1.-qsk), 1./s)

    def _lower(self, k, s):
        return 0.

    def _upper(self, k, s):
        qsk = pow(1-1e-10, s*1./k)
        return pow(qsk/(1.-qsk), 1./s)


class Mielke(ShiftScaleDistribution):
    """
    Mielke's beta-kappa distribution

    Args:
        kappa (float, Distribution):
            First shape parameter
        expo (float, Distribution):
            Second shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Mielke(kappa=1.5, expo=15)
        >>> distribution
        Mielke(1.5, 15)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.342, 0.543, 0.712, 0.868, 3.981])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.877, 1.105, 1.257, 1.234, 0.   ])
        >>> distribution.sample(4).round(3)
        array([0.754, 0.236, 1.028, 0.615])

    """

    def __init__(self, kappa=1, expo=1, scale=1, shift=0):
        super(Mielke, self).__init__(
            dist=mielke(kappa, expo),
            scale=scale,
            shift=shift,
            repr_args=[kappa, expo],
        )
