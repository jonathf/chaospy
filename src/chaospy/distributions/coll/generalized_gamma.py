"""Generalized gamma distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class generalized_gamma(Dist):
    """Generalized gamma distribution."""

    def __init__(self, a, c):
        Dist.__init__(self, a=a, c=c)

    def _pdf(self, x, a, c):
        return abs(c)* numpy.exp((c*a-1)*numpy.log(x)-x**c- special.gammaln(a))

    def _cdf(self, x, a, c):
        val = special.gammainc(a, x**c)
        cond = c + 0*val
        return numpy.where(cond > 0, val, 1-val)

    def _ppf(self, q, a, c):
        val1 = special.gammaincinv(a, q)
        val2 = special.gammaincinv(a, 1.0-q)
        ic = 1.0/c
        cond = c+0*val1
        return numpy.where(cond > 0, val1**ic, val2**ic)

    def _mom(self, k, a, c):
        return special.gamma((c+k)*1./a)/special.gamma(c*1./a)

    def _bnd(self, a, c):
        return 0., self._ppf(1-1e-10, a, c)


class GeneralizedGamma(Add):
    """
    Generalized gamma distribution

    Args:
        shape1 (float, Dist) : Shape parameter 1
        shape2 (float, Dist) : Shape parameter 2
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.GeneralizedGamma(3, 2, 2, 2)
        >>> print(distribution)
        GeneralizedGamma(scale=2, shape1=3, shape2=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [4.4779 5.0233 5.5244 6.1372]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.3145 0.4016 0.3807 0.2624]
        >>> print(numpy.around(distribution.sample(4), 4))
        [5.6691 4.1674 7.0214 5.2264]
        >>> print(numpy.around(distribution.mom(1), 4))
        3.477
    """

    def __init__(self, shape1, shape2, scale, shift):
        self._repr = {
            "shape1": shape1, "shape2": shape2, "scale": scale, "shift": shift}
        Add.__init__(
            self, left=generalized_gamma(shape1, shape2)*scale, right=shift)
