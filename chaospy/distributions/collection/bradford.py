"""Bradford distribution."""
import numpy

from ..baseclass import SimpleDistribution, LowerUpperDistribution


class bradford(SimpleDistribution):
    """Standard Bradford distribution."""

    def __init__(self, c=1):
        super(bradford, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return  c/(c*x+1.0)/numpy.log(1.0+c)

    def _cdf(self, x, c):
        return numpy.log(1.0+c*x)/numpy.log(c+1.0)

    def _ppf(self, q, c):
        return ((1.0+c)**q-1)/c

    def _lower(self, c):
        return 0

    def _upper(self, c):
        return 1


class Bradford(LowerUpperDistribution):
    """
    Bradford distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        lower (float, Distribution):
            Location of lower threshold
        upper (float, Distribution):
            Location of upper threshold

    Examples:
        >>> distribution = chaospy.Bradford(0.8, 4, 6)
        >>> distribution
        Bradford(0.8, lower=4, upper=6)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([4.    , 4.3957, 4.8541, 5.385 , 6.    ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.6805, 0.5875, 0.5072, 0.4379, 0.3781])
        >>> distribution.sample(4).round(4)
        array([5.171 , 4.1748, 5.8704, 4.8192])
        >>> distribution.mom(1).round(4)
        4.9026
    """
    def __init__(self, shape=1, lower=0, upper=1):
        super(Bradford, self).__init__(
            dist=bradford(shape),
            lower=lower,
            upper=upper,
            repr_args=[shape],
        )
