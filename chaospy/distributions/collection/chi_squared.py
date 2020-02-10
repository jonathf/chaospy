"""Non-central Chi-squared distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add
from .deprecate import deprecation_warning


class chi_squared(Dist):
    """Central Chi-squared distribution."""

    def __init__(self, df, nc):
        Dist.__init__(self, df=df, nc=nc)

    def _pdf(self, x, df, nc):
        output = 0.5*numpy.e**(-0.5*(x+nc))
        output *= (x/nc)**(0.25*df-0.5)
        output *= special.iv(0.5*df-1, (nc*x)**0.5)
        return output

    def _cdf(self, x, df, nc):
        return special.chndtr(x, df, nc)

    def _ppf(self, q, df, nc):
        return special.chndtrix(q, df, nc)

    def _lower(self, df, nc):
        return 0.

    def _upper(self, df, nc):
        for expon in range(20, -1, -1):
            upper = self._ppf(1-10**-expon, df, nc)
            if not numpy.isnan(upper).item():
                return upper.item()


class ChiSquared(Add):
    """
    (Non-central) Chi-squared distribution.

    Args:
        df (float, Dist):
            Degrees of freedom
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter
        nc (float, Dist):
            Non-centrality parameter

    Examples:
        >>> distribution = chaospy.ChiSquared(2, 4, 1, 1)
        >>> distribution
        ChiSquared(df=2, nc=1, scale=4, shift=1)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> distribution.inv(q).round(4)
        array([ 3.369 ,  6.1849,  9.7082, 14.5166, 22.4295])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.1667, 0.3333, 0.5   , 0.6667, 0.8333])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.065 , 0.0536, 0.0414, 0.0286, 0.0149])
        >>> distribution.sample(4).round(4)
        array([14.0669,  2.595 , 35.6294,  9.2851])
        >>> distribution.mom(1).round(4)
        13.0001
        >>> distribution.ttr([1, 2, 3]).round(4)
        array([[ 33.    ,  51.6667,  69.8042],
               [128.    , 432.    , 895.9999]])
    """

    def __init__(self, df=1, scale=1, shift=0, nc=0):
        self._repr = {"df": df, "scale": scale, "shift": shift, "nc": nc}
        Add.__init__(self, left=chi_squared(df, nc)*scale, right=shift)


Chisquared = deprecation_warning(ChiSquared, "Chisquared")
