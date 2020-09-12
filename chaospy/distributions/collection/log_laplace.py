"""Log-laplace distribution."""
import numpy
from scipy import special, misc

from ..baseclass import DistributionCore, ShiftScale


class log_laplace(DistributionCore):
    """Log-laplace distribution."""

    def __init__(self, c):
        super(log_laplace, self).__init__(c=c)

    def _pdf(self, x, c):
        cd2 = c/2.0
        c = numpy.where(x < 1, c, -c)
        return cd2*x**(c-1)

    def _cdf(self, x, c):
        return numpy.where(x < 1, 0.5*x**c, 1-0.5*x**(-c))

    def _ppf(self, q, c):
        return numpy.where(q < 0.5, (2.*q)**(1./c), (2*(1.-q))**(-1./c))

    def _lower(self, c):
        return 0.0

class LogLaplace(ShiftScale):
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
        >>> distribution = chaospy.LogLaplace(2, 2, 2)
        >>> distribution
        LogLaplace(2, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([3.2649, 3.7889, 4.2361, 5.1623])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.3162, 0.4472, 0.3578, 0.1265])
        >>> distribution.sample(4).round(4)
        array([4.4028, 2.9592, 8.3425, 3.9641])
    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(LogLaplace, self).__init__(
            dist=log_laplace(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
