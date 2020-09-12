"""(Non-central) F distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class f(DistributionCore):
    """F distribution."""

    def __init__(self, dfn, dfd, nc):
        super(f, self).__init__(dfn=dfn, dfd=dfd, nc=nc)

    def _pdf(self, x, dfn, dfd, nc):
        n1, n2 = dfn, dfd
        term = -nc/2.+nc*n1*x/(2*(n2+n1*x)) + special.gammaln(n1/2.)+special.gammaln(1+n2/2.)
        term -= special.gammaln((n1+n2)/2.)
        Px = numpy.exp(term)
        Px *= n1**(n1/2.) * n2**(n2/2.) * x**(n1/2.-1)
        Px *= (n2+n1*x)**(-(n1+n2)/2.)
        Px *= special.assoc_laguerre(-nc*n1*x/(2.*(n2+n1*x)), n2/2., n1/2.-1)
        Px /= special.beta(n1/2., n2/2.)
        return Px

    def _cdf(self, x, dfn, dfd, nc):
        return special.ncfdtr(dfn, dfd, nc, x)

    def _ppf(self, q, dfn, dfd, nc):
        return special.ncfdtri(dfn, dfd, nc, q)

    def _lower(self, dfn, dfd, nc):
        return 0.


class F(ShiftScale):
    """
    (Non-central) F or Fisher-Snedecor distribution.

    Args:
        n (float, Distribution):
            Degres of freedom for numerator
        m (float, Distribution):
            Degres of freedom for denominator
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter
        nc (float, Distribution):
            Non-centrality parameter

    Examples:
        >>> distribution = chaospy.F(3, 3, 1, scale=2, shift=1)
        >>> distribution
        F(3, 3, nc=1, scale=2, shift=1)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([1.9336, 2.9751, 4.7028, 8.8521])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.2277, 0.1572, 0.0837, 0.027 ])
        >>> distribution.sample(4).round(4)
        array([ 5.4212,  1.5739, 25.7656,  3.5586])
        >>> distribution.mom(1) > 10**8  # undefined
        True
    """

    def __init__(self, n=1, m=1, nc=0, shift=0, scale=1):
        super(F, self).__init__(
            dist=f(n, m, nc),
            shift=shift,
            scale=scale,
            repr_args=[n, m, "nc=%s" % nc],
        )
