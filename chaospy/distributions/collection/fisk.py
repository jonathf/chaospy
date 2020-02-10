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

    def _lower(self, c):
        return 0.


class Fisk(Add):
    """
    Fisk or Log-logistic distribution.

    Args:
        shape (float, Dist):
            Shape parameter
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.Fisk(3, 2, 1)
        >>> distribution
        Fisk(scale=2, shape=3, shift=1)
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
        3.4184
    """
    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=fisk(shape)*scale, right=shift)
