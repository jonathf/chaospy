"""Generalized gamma distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add
from .deprecate import deprecation_warning


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

    def _lower(self, a, c):
        return 0.


class GeneralizedGamma(Add):
    """
    Generalized gamma distribution

    Args:
        shape1 (float, Dist):
            Shape parameter 1
        shape2 (float, Dist):
            Shape parameter 2
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.GeneralizedGamma(3, 2, 2, 2)
        >>> distribution
        GeneralizedGamma(scale=2, shape1=3, shape2=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([4.4779, 5.0233, 5.5244, 6.1372])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.3145, 0.4016, 0.3807, 0.2624])
        >>> distribution.sample(4).round(4)
        array([5.6691, 4.1674, 7.0214, 5.2264])
        >>> distribution.mom(1).round(4)
        3.477
        >>> distribution.ttr([1, 2, 3]).round(4)
        array([[5.6341, 5.9361, 6.2271],
               [0.9553, 1.8381, 2.6689]])
    """

    def __init__(self, shape1, shape2, scale, shift):
        self._repr = {
            "shape1": shape1, "shape2": shape2, "scale": scale, "shift": shift}
        Add.__init__(
            self, left=generalized_gamma(shape1, shape2)*scale, right=shift)


Gengamma = deprecation_warning(GeneralizedGamma, "Gengamma")
