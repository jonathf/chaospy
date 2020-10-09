"""Tukey-lambda distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class tukey_lambda(SimpleDistribution):
    """Tukey-lambda distribution."""

    def __init__(self, lam):
        super(tukey_lambda, self).__init__(dict(lam=lam))

    def _upper(self, lam):
        return 1./numpy.clip(lam, a_min=0.03, a_max=None)

    def _lower(self, lam):
        return -1./numpy.clip(lam, a_min=0.03, a_max=None)

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
        lam = numpy.broadcast_to(lam, q.shape)
        indices = lam != 0
        q_ = q[indices]
        lam_ = lam[indices]
        output[indices] = (q_**lam_ - (1-q_)**lam_)/lam_
        q_ = q[~indices]
        output[~indices] = numpy.log(q_/(1-q_))
        return output


class TukeyLambda(ShiftScaleDistribution):
    """
    Tukey-lambda distribution.

    Args:
        lam (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.TukeyLambda(1.5)
        >>> distribution
        TukeyLambda(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-0.667, -0.417, -0.141,  0.141,  0.417,  0.667])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.745, 0.711, 0.711, 0.745, 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.216, -0.529,  0.61 , -0.025])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(TukeyLambda, self).__init__(
            dist=tukey_lambda(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
