"""Log-Normal probability distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class log_normal(Dist):

    def __init__(self, a=1):
        Dist.__init__(self, a=a)

    def _pdf(self, x, a):
        out = numpy.e**(-numpy.log(x+(1-x)*(x<=0))**2/(2*a*a)) / \
            ((x+(1-x)*(x<=0))*a*numpy.sqrt(2*numpy.pi))*(x>0)
        return out

    def _cdf(self, x, a):
        return special.ndtr(numpy.log(x+(1-x)*(x<=0))/a)*(x>0)

    def _ppf(self, x, a):
        return numpy.e**(a*special.ndtri(x))

    def _mom(self, k, a):
        return numpy.e**(.5*a*a*k*k)

    def _ttr(self, n, a):
        return \
    (numpy.e**(n*a*a)*(numpy.e**(a*a)+1)-1)*numpy.e**(.5*(2*n-1)*a*a), \
                (numpy.e**(n*a*a)-1)*numpy.e**((3*n-2)*a*a)

    def _bnd(self, x, a):
        return 0, self._ppf(1-1e-10, a)


class LogNormal(Add):
    R"""
    Log-normal distribution

    Args:
        mu (float, Dist) : Mean in the normal distribution.  Overlaps with
                scale by mu=log(scale)
        sigma (float, Dist) : Standard deviation of the normal distribution.
        shift (float, Dist) : Location of the lower bound.
        scale (float, Dist) : Scale parameter. Overlaps with mu by scale=e**mu

    Examples:
        >>> distribution = chaospy.LogNormal(0, 1)
        >>> print(distribution)
        LogNormal(mu=0, scale=1, shift=0, sigma=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.431  0.7762 1.2883 2.3201]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.6495 0.4977 0.2999 0.1207]
        >>> print(numpy.around(distribution.sample(4), 4))
        [1.4844 0.3011 5.1945 0.9563]
        >>> print(numpy.around(distribution.mom(1), 4))
        1.6487
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[1.50155000e+01 1.18650900e+02 8.97651100e+02]
         [4.67080000e+00 3.48830600e+02 2.09298326e+04]]
    """

    def __init__(self, mu=0, sigma=1, shift=0, scale=1):
        self._repr = {"mu": mu, "sigma": sigma, "shift": shift, "scale": scale}
        left = log_normal(sigma)*scale*numpy.e**mu
        Add.__init__(self, left=left, right=shift)


class Gilbrat(Add):
    """
    Gilbrat distribution.

    Standard log-normal distribution

    Args:
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.Gilbrat(3, 2)
        >>> print(distribution)
        Gilbrat(scale=3, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [3.293  4.3286 5.865  8.9604]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.2165 0.1659 0.1    0.0402]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 6.4533  2.9033 17.5835  4.869 ]
        >>> print(numpy.around(distribution.mom(1), 4))
        6.9462
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[4.70464000e+01 3.57952700e+02 2.69495320e+03]
         [4.20370000e+01 3.13947580e+03 1.88368494e+05]]
    """
    def __init__(self, scale=1, shift=0):
        self._repr = {"scale": scale, "shift": shift}
        Add.__init__(self, left=log_normal(1)*scale, right=shift)
