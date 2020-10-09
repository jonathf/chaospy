"""Nakagami-m distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class nakagami(SimpleDistribution):
    """Nakagami-m distribution."""

    def __init__(self, nu):
        super(nakagami, self).__init__(dict(nu=nu))

    def _pdf(self, x, nu):
        return 2*nu**nu/special.gamma(nu)*(x**(2*nu-1.0))*numpy.exp(-nu*x*x)

    def _cdf(self, x, nu):
        return special.gammainc(nu,nu*x*x)

    def _ppf(self, q, nu):
        return numpy.sqrt(1.0/nu*special.gammaincinv(nu, q))

    def _lower(self, nu):
        return 0.

    def _upper(self, nu):
        return numpy.sqrt(1.0/nu*special.gammaincinv(nu, 1-1e-16))


class Nakagami(ShiftScaleDistribution):
    """
    Nakagami-m distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Nakagami(1.5)
        >>> distribution
        Nakagami(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.579, 0.789, 0.991, 1.244, 5.079])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.84 , 1.015, 0.933, 0.63 , 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.05 , 0.465, 1.615, 0.87 ])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Nakagami, self).__init__(
            dist=nakagami(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
