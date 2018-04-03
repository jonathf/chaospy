"""Log-laplace distribution."""
import numpy
from scipy import special, misc

from ..baseclass import Dist
from ..operators.addition import Add


class log_laplace(Dist):
    """Log-laplace distribution."""

    def __init__(self, c):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        cd2 = c/2.0
        c = numpy.where(x < 1, c, -c)
        return cd2*x**(c-1)

    def _cdf(self, x, c):
        return numpy.where(x < 1, 0.5*x**c, 1-0.5*x**(-c))

    def _ppf(self, q, c):
        return numpy.where(q < 0.5, (2.*q)**(1./c), (2*(1.-q))**(-1./c))

    def _bnd(self, x, c):
        return 0.0, self._ppf(1-1e-10, c)

class LogLaplace(Add):
    """
    Log-laplace distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> f = chaospy.LogLaplace(2, 2, 2)
        >>> print(f)
        LogLaplace(scale=2, shape=2, shift=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [3.2649 3.7889 4.2361 5.1623]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.pdf(f.inv(q)), 4))
        [0.3162 0.4472 0.3578 0.1265]
        >>> print(numpy.around(f.sample(4), 4))
        [4.4028 2.9592 8.3425 3.9641]
        >>> print(numpy.around(f.mom(1), 4))
        70712.6752
    """

    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=log_laplace(shape)*scale, right=shift)
