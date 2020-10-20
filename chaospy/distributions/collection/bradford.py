"""Bradford distribution."""
import numpy

from ..baseclass import SimpleDistribution, LowerUpperDistribution


class bradford(SimpleDistribution):
    """Standard Bradford distribution."""

    def __init__(self, c=1):
        super(bradford, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return  c/(c*x+1.)/numpy.log(1.+c)

    def _cdf(self, x, c):
        return numpy.log(1.+c*x)/numpy.log(c+1.)

    def _ppf(self, q, c):
        return ((1.+c)**q-1)/c

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return 1.


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
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([4.   , 4.312, 4.663, 5.057, 5.501, 6.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.681, 0.605, 0.538, 0.478, 0.425, 0.378])
        >>> distribution.sample(4).round(3)
        array([5.171, 4.175, 5.87 , 4.819])

    """
    def __init__(self, shape=1, lower=0, upper=1):
        super(Bradford, self).__init__(
            dist=bradford(shape),
            lower=lower,
            upper=upper,
            repr_args=[shape],
        )
