"""Tukey-lambda distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add
from .deprecate import deprecation_warning


class tukey_lambda(Dist):
    """Tukey-lambda distribution."""

    def __init__(self, lam):
        Dist.__init__(self, lam=lam)

    def _pdf(self, x, lam):
        lam = numpy.zeros(x.shape) + lam
        output = numpy.zeros(x.shape)
        indices = (lam <= 0) | (numpy.abs(x)*lam < 1)
        lam = lam[indices]
        Fx = special.tklmbda(x[indices], lam)
        Px = 1/(Fx**(lam-1.0) + ((1-Fx))**(lam-1.0))
        output[indices] = Px
        return output

    def _cdf(self, x, lam):
        return special.tklmbda(x, lam)

    def _ppf(self, q, lam):
        output = numpy.zeros(q.shape)
        lam = numpy.zeros(q.shape) + lam
        indices = lam != 0
        q_ = q[indices]
        lam_ = lam[indices]
        output[indices] = (q_**lam_ - (1-q_)**lam_)/lam_
        q_ = q[~indices]
        output[~indices] = numpy.log(q_/(1-q_))
        return output


class TukeyLambda(Add):
    """
    Tukey-lambda distribution.

    Args:
        lam (float, Dist):
            Shape parameter
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.TukeyLambda(0, 2, 2)
        >>> distribution
        TukeyLambda(scale=2, shape=0, shift=2)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> distribution.inv(q).round(4)
        array([-1.2189,  0.6137,  2.    ,  3.3863,  5.2189])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.1667, 0.3333, 0.5   , 0.6667, 0.8333])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0694, 0.1111, 0.125 , 0.1111, 0.0694])
        >>> distribution.sample(4).round(4)
        array([ 3.2697, -2.0812,  7.9008,  1.8575])
        >>> distribution.mom(1).round(4)
        2.0
    """

    def __init__(self, shape=0, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=tukey_lambda(shape)*scale, right=shift)


Tukeylambda = deprecation_warning(TukeyLambda, "Tukeylambda")
