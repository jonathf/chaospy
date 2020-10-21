"""Non-central Chi-squared distribution."""
import numpy
from scipy import special
import chaospy

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

    def _ppf(self, qloc, df, nc):
        qloc = numpy.clip(qloc, None, 1-1e-12)
        return special.chndtrix(qloc, df, nc)

    def _lower(self, df, nc):
        return 0.

    def _upper(self, df, nc):
        return special.chndtrix(1-1e-12, df, nc)


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
        >>> distribution = chaospy.ChiSquared(df=15)
        >>> distribution
        ChiSquared(15)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([ 0.   , 11.003, 13.905, 16.784, 20.592, 95.358])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.062, 0.072, 0.064, 0.041, 0.   ])
        >>> distribution.sample(4).round(3)
        array([17.655,  9.454, 26.66 , 15.047])

    """

    def __init__(self, df=1, nc=1, scale=1, shift=0):
        super(ChiSquared, self).__init__(
            dist=chi_squared(df, nc),
            scale=scale,
            shift=shift,
            repr_args=[df]+chaospy.format_repr_kwargs(nc=(nc, 1)),
        )
