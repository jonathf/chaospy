"""Mielke's beta-kappa distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class mielke(DistributionCore):
    """Mielke's beta-kappa distribution."""

    def __init__(self, k, s):
        super(mielke, self).__init__(k=k, s=s)

    def _pdf(self, x, k, s):
        return k*x**(k-1.0)/(1.0+x**s)**(1.0+k*1.0/s)

    def _cdf(self, x, k, s):
        return x**k/(1.0+x**s)**(k*1.0/s)

    def _ppf(self, q, k, s):
        qsk = pow(q,s*1.0/k)
        return pow(qsk/(1.0-qsk),1.0/s)

    def _lower(self, k, s):
        return 0.


class Mielke(ShiftScale):
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
        >>> distribution = chaospy.Mielke(2, 0.5, scale=2)
        >>> distribution
        Mielke(2, 0.5, scale=2)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> distribution.inv(q).round(4)
        array([  6.2633,  20.0195,  55.867 , 175.731 , 919.6095])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.1667, 0.3333, 0.5   , 0.6667, 0.8333])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0192, 0.008 , 0.0028, 0.0007, 0.0001])
        >>> distribution.sample(4).round(4)
        array([1.58937400e+02, 3.88830000e+00, 1.21490448e+04, 4.99808000e+01])
    """

    def __init__(self, kappa=1, expo=1, scale=1, shift=0):
        super(Mielke, self).__init__(
            dist=mielke(kappa, expo),
            scale=scale,
            shift=shift,
            repr_args=[kappa, expo],
        )
