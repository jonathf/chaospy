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
        >>> distribution = chaospy.Fisk(3, 2, 1)
        >>> distribution
        Fisk(3, scale=2, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.2599, 2.7472, 3.2894, 4.1748])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.381 , 0.4121, 0.3145, 0.1512])
        >>> distribution.sample(4).round(4)
        array([3.4714, 2.013 , 6.3474, 2.9531])
        >>> distribution.mom(1).round(4)
        3.5577

    """
    def __init__(self, shape=1, scale=1, shift=0):
        super(Fisk, self).__init__(
            dist=fisk(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
