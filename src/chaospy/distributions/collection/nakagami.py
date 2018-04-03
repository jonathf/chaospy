"""Nakagami-m distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class nakagami(Dist):
    """Nakagami-m distribution."""

    def __init__(self, nu):
        Dist.__init__(self, nu=nu)

    def _pdf(self, x, nu):
        return 2*nu**nu/special.gamma(nu)*(x**(2*nu-1.0))*numpy.exp(-nu*x*x)

    def _cdf(self, x, nu):
        return special.gammainc(nu,nu*x*x)

    def _ppf(self, q, nu):
        return numpy.sqrt(1.0/nu*special.gammaincinv(nu, q))

    def _bnd(self, x, nu):
        return 0.0, self._ppf(1-1e-10, nu)


class Nakagami(Add):
    """
    Nakagami-m distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> f = chaospy.Nakagami(2, 2, 2)
        >>> print(f)
        Nakagami(scale=2, shape=2, shift=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [3.284  3.6592 4.0111 4.4472]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.pdf(f.inv(q)), 4))
        [0.4642 0.5766 0.5383 0.3669]
        >>> print(numpy.around(f.sample(4), 4))
        [4.1137 3.076  5.0824 3.8012]
        >>> print(numpy.around(f.mom(1), 4))
        5.6286
    """

    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, nakagami(shape)*scale, shift)
