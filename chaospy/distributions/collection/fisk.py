"""Fisk or Log-logistic distribution."""
from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class fisk(SimpleDistribution):
    """Fisk or Log-logistic distribution."""

    def __init__(self, c=1):
        super(fisk, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return c*x**(c-1.)/(1+x**c)**2

    def _cdf(self, x, c):
        return 1./(1+x**-c)

    def _ppf(self, q, c):
        return (q**(-1.0)-1)**(-1.0/c)

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return 1e12**(1./c)


class Fisk(ShiftScaleDistribution):
    """
    Fisk or Log-logistic distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Fisk(5)
        >>> distribution
        Fisk(5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([  0.   ,   0.758,   0.922,   1.084,   1.32 , 251.189])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 1.056, 1.301, 1.107, 0.606, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.135, 0.665, 1.804, 0.986])

    """
    def __init__(self, shape=1, scale=1, shift=0):
        super(Fisk, self).__init__(
            dist=fisk(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
