"""Generalized exponential distribution."""
import numpy
from scipy import special
import chaospy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class generalized_exponential(SimpleDistribution):
    """Generalized exponential distribution."""

    def __init__(self, a=1, b=1, c=1):
        super(generalized_exponential, self).__init__(dict(a=a, b=b, c=c))

    def _pdf(self, x, a, b, c):
        return (a+b*(-numpy.expm1(-c*x)))*numpy.exp((-a-b)*x+b*(-numpy.expm1(-c*x))/c)

    def _cdf(self, x, a, b, c):
        output = -numpy.expm1((-a-b)*x + b*(-numpy.expm1(-c*x))/c)
        output = numpy.where(x > 0, output, 0)
        return output

    def _lower(self, a, b, c):
        return 0.

    def _upper(self, a, b, c):
        qloc, a, b, c = numpy.broadcast_arrays(1-1e-12, a, b, c)
        return chaospy.approximate_inverse(
            distribution=self,
            idx=0,
            qloc=qloc,
            parameters=dict(a=a, b=b, c=c),
            bounds=(0., 500./a/b/c),
            tolerance=1e-15,
        )


class GeneralizedExponential(ShiftScaleDistribution):
    """
    Generalized exponential distribution.

    Args:
        a (float, Distribution):
            First shape parameter
        b (float, Distribution):
            Second shape parameter
        c (float, Distribution):
            Third shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Note:
        "An Extension of Marshall and Olkin's Bivariate Exponential Distribution",
        H.K. Ryu, Journal of the American Statistical Association, 1993.

        "The Exponential Distribution: Theory, Methods and Applications",
        N. Balakrishnan, Asit P. Basu.

    Examples:
        >>> distribution = chaospy.GeneralizedExponential(3, 4, 5)
        >>> distribution
        GeneralizedExponential(3, 4, 5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.063, 0.127, 0.204, 0.321, 4.062])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([3.   , 3.26 , 2.925, 2.223, 1.24 , 0.   ])
        >>> distribution.sample(4).round(3)
        array([0.08 , 0.316, 0.318, 0.003])

    """

    def __init__(self, a=1, b=1, c=1, scale=1, shift=0):
        super(GeneralizedExponential, self).__init__(
            dist=generalized_exponential(a, b, c),
            scale=scale,
            shift=shift,
            repr_args=[a, b, c],
        )
