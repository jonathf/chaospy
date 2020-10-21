"""Reciprocal distribution."""
import numpy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class reciprocal(SimpleDistribution):

    def __init__(self, lower=1, upper=2):
        super(reciprocal, self).__init__(dict(lower=lower, upper=upper))

    def _pdf(self, x, lower, upper):
        return 1./(x*numpy.log(upper/lower))

    def _cdf(self, x, lower, upper):
        return numpy.log(x/lower)/numpy.log(upper/lower)

    def _ppf(self, q, lower, upper):
        return numpy.e**(q*numpy.log(upper/lower)+numpy.log(lower))

    def _lower(self, lower, upper):
        return lower

    def _upper(self, lower, upper):
        return upper

    def _mom(self, kloc, lower, upper):
        return (upper**kloc-lower**kloc)/(kloc*numpy.log(upper/lower))


class Reciprocal(ShiftScaleDistribution):
    """
    Reciprocal distribution.

    Args:
        lower (float, Distribution):
            Lower threshold of distribution. Must be smaller than ``upper``.
        upper (float, Distribution):
            Upper threshold of distribution.
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter


    Examples:
        >>> distribution = chaospy.Reciprocal(2, 4)
        >>> distribution
        Reciprocal(2, 4)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([2.   , 2.297, 2.639, 3.031, 3.482, 4.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.721, 0.628, 0.547, 0.476, 0.414, 0.361])
        >>> distribution.sample(4).round(3)
        array([3.146, 2.166, 3.865, 2.794])
        >>> distribution.mom(1).round(4)
        2.8854

    """

    def __init__(self, lower, upper, shift=0, scale=1):
        super(Reciprocal, self).__init__(
            dist=reciprocal(lower, upper),
            shift=shift, scale=scale,
            repr_args=[lower, upper],
        )
