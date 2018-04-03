"""Mielke's beta-kappa distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class mielke(Dist):
    """Mielke's beta-kappa distribution."""

    def __init__(self, k, s):
        Dist.__init__(self, k=k, s=s)

    def _pdf(self, x, k, s):
        return k*x**(k-1.0) / (1.0+x**s)**(1.0+k*1.0/s)

    def _cdf(self, x, k, s):
        return x**k / (1.0+x**s)**(k*1.0/s)

    def _ppf(self, q, k, s):
        qsk = pow(q,s*1.0/k)
        return pow(qsk/(1.0-qsk),1.0/s)

    def _bnd(self, x, k, s):
        return 0.0, self._ppf(1-1e-10, k, s)


class Mielke(Add):
    """
    Mielke's beta-kappa distribution

    Args:
        kappa (float, Dist) : First shape parameter
        expo (float, Dist) : Second shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.Mielke(2, 0.5, 2)
        >>> print(distribution)
        Mielke(expo=0.5, kappa=2, scale=2, shift=0)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [  6.2633  20.0195  55.867  175.731  919.6095]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.0192 0.008  0.0028 0.0007 0.0001]
        >>> print(numpy.around(distribution.sample(4), 4))
        [1.58937400e+02 3.88830000e+00 1.21490448e+04 4.99808000e+01]
    """

    def __init__(self, kappa=1, expo=1, scale=1, shift=0):
        self._repr = {
            "kappa": kappa, "expo": expo, "scale": scale, "shift": shift}
        Add.__init__(self, left=mielke(kappa, expo)*scale, right=shift)
