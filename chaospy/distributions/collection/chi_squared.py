"""Non-central Chi-squared distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class chi_squared(SimpleDistribution):
    """Central Chi-squared distribution."""

    def __init__(self, df, nc):
        super(chi_squared, self).__init__(dict(df=df, nc=nc))

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


class ChiSquared(ShiftScaleDistribution):
    """
    (Non-central) Chi-squared distribution.

    Args:
        df (float, Distribution):
            Degrees of freedom
        nc (float, Distribution):
            Non-centrality parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.ChiSquared(2, 1, scale=4, shift=1)
        >>> distribution
        ChiSquared(2, nc=1, scale=4, shift=1)
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
    """

    def __init__(self, df=1, nc=0, scale=1, shift=0):
        super(ChiSquared, self).__init__(
            dist=chi_squared(df, nc),
            scale=scale,
            shift=shift,
            repr_args=[df, "nc=%s" % nc],
        )
