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

    def _mom(self, k, lower, upper):
        return ((upper*numpy.e**k-lower*numpy.e**k)/(numpy.log(upper/lower)*(k+(k == 0))))**(k != 0)


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
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([2.    , 2.3784, 2.8284, 3.3636, 4.    ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.7213, 0.6066, 0.5101, 0.4289, 0.3607])
        >>> distribution.sample(4).round(4)
        array([3.1462, 2.166 , 3.8645, 2.7937])
        >>> distribution.mom(1).round(4)
        7.8433
    """

    def __init__(self, lower, upper, shift=0, scale=1):
        super(Reciprocal, self).__init__(
            dist=reciprocal(lower, upper),
            shift=shift, scale=scale,
            repr_args=[lower, upper],
        )
