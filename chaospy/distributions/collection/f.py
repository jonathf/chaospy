"""(Non-central) F distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class f(SimpleDistribution):
    """F distribution."""

    def __init__(self, dfn, dfd, nc):
        super(f, self).__init__(
            parameters=dict(dfn=dfn, dfd=dfd, nc=nc),
            repr_args=[dfn, dfd, "nc=%s" % nc],
        )

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
        return numpy.where(q == 1, self._upper(dfn, dfd, nc), special.ncfdtri(dfn, dfd, nc, q))

    def _lower(self, dfn, dfd, nc):
        return 0.

    def _upper(self, dfn, dfd, nc):
        return special.ncfdtri(dfn, dfd, nc, 1-1e-10)


class F(ShiftScaleDistribution):
    """
    (Non-central) F or Fisher-Snedecor distribution.

    Args:
        n (float, Distribution):
            Degres of freedom for numerator
        m (float, Distribution):
            Degres of freedom for denominator
        nc (float, Distribution):
            Non-centrality parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.F(10, 10, 0)
        >>> distribution
        F(10, 10, nc=0)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([  0.   ,   0.578,   0.848,   1.179,   1.732, 261.403])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.734, 0.701, 0.505, 0.245, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.292, 0.455, 2.984, 0.971])

    """

    def __init__(self, n=2, m=10, nc=0, shift=0, scale=1):
        super(F, self).__init__(
            dist=f(n, m, nc),
            shift=shift,
            scale=scale,
            repr_args=[n, m, "nc=%s" % nc],
        )
