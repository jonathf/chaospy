"""Fisk or Log-logistic distribution."""
from ..baseclass import Dist
from ..operators.addition import Add


class fisk(Dist):
    """Fisk or Log-logistic distribution."""

    def __init__(self, c=1.):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        output = c*x**(c-1.)
        output /= (1+x**c)**2
        return output

    def _cdf(self, x, c):
        return 1./(1+x**-c)

    def _ppf(self, q, c):
        return (q**(-1.0)-1)**(-1.0/c)

    def _bnd(self, c):
        return 0, self._ppf(1-1e-10, c)


class Fisk(Add):
    """
    Fisk or Log-logistic distribution.

    Args:
        shape (float, Dist): Shape parameter
        scale (float, Dist): Scaling parameter
        shift (float, Dist): Location parameter

    Examples:
        >>> distribution = chaospy.Fisk(3, 2, 1)
        >>> print(distribution)
        Fisk(scale=2, shape=3, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [2.2599 2.7472 3.2894 4.1748]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.381  0.4121 0.3145 0.1512]
        >>> print(numpy.around(distribution.sample(4), 4))
        [3.4714 2.013  6.3474 2.9531]
        >>> print(numpy.around(distribution.mom(1), 4))
        2155.4346
    """
    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=fisk(shape)*scale, right=shift)
