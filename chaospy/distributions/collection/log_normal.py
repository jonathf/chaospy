"""Log-Normal probability distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class log_normal(SimpleDistribution):

    def __init__(self, a=1):
        super(log_normal, self).__init__(dict(a=a))

    def _lower(self, a):
        return 0.

    def _upper(self, a):
        return numpy.e**(a*6.37)

    def _pdf(self, x, a):
        out = (numpy.e**(-numpy.log(x+(1-x)*(x<=0))**2/(2*a*a))/
               ((x+(1-x)*(x<=0))*a*numpy.sqrt(2*numpy.pi))*(x>0))
        return out

    def _cdf(self, x, a):
        return special.ndtr(numpy.log(x+(1-x)*(x<=0))/a)*(x>0)

    def _ppf(self, x, a):
        return numpy.e**(a*special.ndtri(x))

    def _mom(self, k, a):
        return numpy.e**(.5*a*a*k*k)

    def _ttr(self, n, a):
        """Stieltjes-Wigert."""
        return (
            (numpy.e**(n*a*a)*(numpy.e**(a*a)+1)-1)*numpy.e**(.5*(2*n-1)*a*a),
            (numpy.e**(n*a*a)-1)*numpy.e**((3*n-2)*a*a)
        )


class LogNormal(ShiftScaleDistribution):
    R"""
    Log-normal distribution

    Args:
        mu (float, Distribution):
            Mean in the normal distribution.  Overlaps with scale by
            mu=log(scale)
        sigma (float, Distribution):
            Standard deviation of the normal distribution.
        shift (float, Distribution):
            Location of the lower bound.
        scale (float, Distribution):
            Scale parameter. Overlaps with mu by scale=e**mu

    Examples:
        >>> distribution = chaospy.LogNormal(0, 0.1)
        >>> distribution
        LogNormal(mu=0, sigma=0.1)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.919, 0.975, 1.026, 1.088, 1.891])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 3.045, 3.963, 3.767, 2.574, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.04 , 0.887, 1.179, 0.996])
        >>> distribution.mom(1).round(3)
        1.005
        >>> distribution.ttr([0, 1, 2, 3]).round(3)
        array([[1.005, 1.035, 1.067, 1.098],
               [0.   , 0.01 , 0.021, 0.033]])

    """

    def __init__(self, mu=0, sigma=1, shift=0, scale=1):
        dist = ShiftScaleDistribution(dist=log_normal(sigma), scale=numpy.e**mu)
        super(LogNormal, self).__init__(
            dist=dist,
            scale=scale,
            shift=shift,
            repr_args=["mu=%s" % mu, "sigma=%s" % sigma],
        )


class Gilbrat(ShiftScaleDistribution):
    """
    Gilbrat distribution.

    Standard log-normal distribution

    Args:
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Gilbrat(scale=0.0015)
        >>> distribution
        Gilbrat(scale=0.0015)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.001, 0.001, 0.002, 0.003, 0.876])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([  0.   , 433.031, 331.825, 199.919,  80.444,   0.   ])
        >>> distribution.sample(4).round(4)
        array([0.0022, 0.0005, 0.0078, 0.0014])
        >>> distribution.mom(1).round(8)
        0.00247308
        >>> distribution.ttr([0, 1, 2]).round(4)
        array([[0.0025, 0.0225, 0.178 ],
               [0.    , 0.    , 0.0008]])

    """

    def __init__(self, scale=1, shift=0):
        super(Gilbrat, self).__init__(
            dist=log_normal(1),
            scale=scale,
            shift=shift,
            repr_args=[],
        )
