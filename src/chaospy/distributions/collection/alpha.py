"""Alpha probability distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class alpha(Dist):
    """Standard Alpha distribution."""

    def __init__(self, a=1):
        Dist.__init__(self, a=a)

    def _cdf(self, x, a):
        return special.ndtr(a-1./x) / special.ndtr(a)

    def _ppf(self, q, a):
        return 1.0/(a-special.ndtri(q*special.ndtr(a)))

    def _pdf(self, x, a):
        return 1.0/(x**2)/special.ndtr(a)*numpy.e**(.5*(a-1.0/x)**2)/numpy.sqrt(2*numpy.pi)

    def _bnd(self, x, a):
        return 0, self._ppf(1-1e-10, a)


class Alpha(Add):
    """
    Alpha distribution.

    Args:
        shape (float, Dist): Shape parameter
        scale (float, Dist): Scale Parameter
        shift (float, Dist): Location of lower threshold

    Examples:
        >>> distribution = chaospy.Alpha(2, 0.5, 4)
        >>> print(distribution)
        Alpha(scale=0.5, shape=2, shift=4)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [4.1676 4.2039 4.2465 4.3104 4.4521]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [11.7723  5.4345  3.361   2.2848  1.4892]
        >>> print(numpy.around(distribution.sample(4), 4))
        [4.304  4.1556 4.9362 4.2413]
        >>> print(distribution.mom(1) > 1e8) # actually undefined.
        True
    """

    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=alpha(shape)*scale, right=shift)
