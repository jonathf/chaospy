"""(Non-central) F distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class f(Dist):
    """F distribution."""

    def __init__(self, dfn, dfd, nc):
        Dist.__init__(self, dfn=dfn, dfd=dfd, nc=nc)

    def _pdf(self, x, dfn, dfd, nc):
        n1, n2 = dfn, dfd
        term = -nc/2+nc*n1*x/(2*(n2+n1*x)) + special.gammaln(n1/2.)+special.gammaln(1+n2/2.)
        term -= special.gammaln((n1+n2)/2.0)
        Px = numpy.exp(term)
        Px *= n1**(n1/2) * n2**(n2/2) * x**(n1/2-1)
        Px *= (n2+n1*x)**(-(n1+n2)/2)
        Px *= special.assoc_laguerre(-nc*n1*x/(2.0*(n2+n1*x)), n2/2, n1/2-1)
        Px /= special.beta(n1/2, n2/2)
        return Px

    def _cdf(self, x, dfn, dfd, nc):
        return special.ncfdtr(dfn, dfd, nc, x)

    def _ppf(self, q, dfn, dfd, nc):
        return special.ncfdtri(dfn, dfd, nc, q)

    def _bnd(self, x, dfn, dfd, nc):
        return 0.0, self._ppf(1-1e-10, dfn, dfd, nc)


class F(Add):
    """
    (Non-central) F or Fisher-Snedecor distribution.

    Args:
        n (float, Dist) : Degres of freedom for numerator
        m (float, Dist) : Degres of freedom for denominator
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter
        nc (float, Dist) : Non-centrality parameter

    Examples:
        >>> distribution = chaospy.F(3, 3, 2, 1, 1)
        >>> print(distribution)
        F(m=3, n=3, nc=1, scale=2, shift=1)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [1.9336 2.9751 4.7028 8.8521]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.2277 0.1572 0.0837 0.027 ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 5.4212  1.5739 25.7656  3.5586]
        >>> print(distribution.mom(1) > 10**8) # undefined
        True
    """

    def __init__(self, n=1, m=1, scale=1, shift=0, nc=0):
        self._repr = {"n": n, "m": m, "scale": scale, "shift": shift, "nc": nc}
        Add.__init__(self, left=f(n, m, nc)*scale, right=shift)
