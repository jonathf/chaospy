"""Log-laplace distribution."""
import numpy
from scipy import special, misc

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class log_laplace(SimpleDistribution):
    """Log-laplace distribution."""

    def __init__(self, c):
        super(log_laplace, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        cd2 = c/2.0
        c = numpy.where(x < 1, c, -c)
        return cd2*x**(c-1)

    def _cdf(self, x, c):
        return numpy.where(x < 1, 0.5*x**c, 1-0.5*x**(-c))

    def _ppf(self, q, c):
        return numpy.where(q < 0.5, (2.*q)**(1./c), (2*(1.-q))**(-1./c))

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return 2e12**(1./c)


class LogLaplace(ShiftScaleDistribution):
    """
    Log-laplace distribution

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.LogLaplace(5)
        >>> distribution
        LogLaplace(5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([  0.   ,   0.833,   0.956,   1.046,   1.201, 288.54 ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 1.201, 2.091, 1.913, 0.833, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.076, 0.745, 1.587, 0.993])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(LogLaplace, self).__init__(
            dist=log_laplace(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
