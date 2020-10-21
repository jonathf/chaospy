"""Kumaraswswamy's double bounded distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, LowerUpperDistribution


class kumaraswamy(SimpleDistribution):
    """Kumaraswamy's double bounded distribution."""

    def __init__(self, a=1, b=1):
        # assert numpy.all(a > 0) and numpy.all(b > 0)
        super(kumaraswamy, self).__init__(dict(a=a, b=b))

    def _pdf(self, x, a, b):
        return a*b*x**(a-1)*(1-x**a)**(b-1)

    def _cdf(self, x, a, b):
        return 1-(1-x**a)**b

    def _ppf(self, q, a, b):
        return (1-(1-q)**(1./b))**(1./a)

    def _mom(self, k, a, b):
        return (b*special.gamma(1+k*1./a)*special.gamma(b)/
                special.gamma(1+b+k*1./a))

    def _lower(self, a, b):
        return 0.

    def _upper(self, a, b):
        return 1.


class Kumaraswamy(LowerUpperDistribution):
    """
    Kumaraswamy's double bounded distribution

    Args:
        alpha (float, Distribution):
            First shape parameter, alpha > 0
        beta (float, Distribution):
            Second shape parameter, b > 0
        lower (float, Distribution):
            Lower threshold
        upper (float, Distribution):
            Upper threshold

    Examples:
        >>> distribution = chaospy.Kumaraswamy(1.5, 2.5)
        >>> distribution
        Kumaraswamy(1.5, 2.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.194, 0.324, 0.455, 0.609, 1.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 1.444, 1.572, 1.46 , 1.114, 0.   ])
        >>> distribution.sample(4).round(3)
        array([0.492, 0.132, 0.788, 0.377])
        >>> distribution.mom(1).round(4)
        0.404

    """

    def __init__(self, alpha, beta, lower=0, upper=1):
        super(Kumaraswamy, self).__init__(
            dist=kumaraswamy(alpha, beta),
            lower=lower,
            upper=upper,
            repr_args=[alpha, beta],
        )
