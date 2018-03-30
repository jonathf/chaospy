"""Tukey-lambda distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class tukey_lambda(Dist):
    """Tukey-lambda distribution."""

    def __init__(self, lam):
        Dist.__init__(self, lam=lam)

    def _pdf(self, x, lam):
        Fx = (special.tklmbda(x, lam))
        Px = Fx**(lam-1.0) + ((1-Fx))**(lam-1.0)
        Px = 1.0/(Px)
        return numpy.where((lam <= 0) | (abs(x) < 1.0/(lam)), Px, 0.0)

    def _cdf(self, x, lam):
        return special.tklmbda(x, lam)

    def _ppf(self, q, lam):
        q = q*1.0
        vals1 = (q**lam - (1-q)**lam)/lam
        vals2 = numpy.log(q/(1-q))
        return numpy.where((lam==0)&(q==q), vals2, vals1)

    def _bnd(self, lam):
        return self._ppf(1e-10, lam), self._ppf(1-1e-10, lam)


class TukeyLambda(Add):
    """
    Tukey-lambda distribution.

    Args:
        lam (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.TukeyLambda(0, 2, 2)
        >>> print(distribution)
        TukeyLambda(scale=2, shape=0, shift=2)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [-1.2189  0.6137  2.      3.3863  5.2189]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.0694 0.1111 0.125  0.1111 0.0694]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 3.2697 -2.0812  7.9008  1.8575]
        >>> print(numpy.around(distribution.mom(1), 4))
        2.0
    """

    def __init__(self, shape=0, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=tukey_lambda(shape)*scale, right=shift)
