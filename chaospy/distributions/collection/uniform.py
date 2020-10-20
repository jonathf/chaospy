"""Uniform probability distribution."""
from ..baseclass import SimpleDistribution, LowerUpperDistribution


class uniform(SimpleDistribution):
    """Uniform distribution fixed on the [-1, 1] interval."""

    def __init__(self):
        super(uniform, self).__init__()

    def _pdf(self, x):
        return 0.5

    def _cdf(self, x):
        return .5*x+.5

    def _ppf(self, q):
        return 2*q-1

    def _lower(self):
        return -1.

    def _upper(self):
        return 1.

    def _mom(self, k):
        return 1./(k+1)*(k%2 == 0)

    def _ttr(self, n):
        return 0., n*n/(4.*n*n-1)


class Uniform(LowerUpperDistribution):
    r"""
    Uniform probability distribution.

    Args:
        lower (float, Distribution):
            Lower threshold of distribution. Must be smaller than ``upper``.
        upper (float, Distribution):
            Upper threshold of distribution.

    Examples:
        >>> distribution = chaospy.Uniform(2, 4)
        >>> distribution
        Uniform(lower=2, upper=4)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([2. , 2.4, 2.8, 3.2, 3.6, 4. ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        >>> distribution.sample(4).round(3)
        array([3.307, 2.23 , 3.901, 2.964])
        >>> distribution.mom(1).round(4)
        3.0
        >>> distribution.ttr([0, 1, 2, 3]).round(4)
        array([[ 3.    ,  3.    ,  3.    ,  3.    ],
               [-0.    ,  0.3333,  0.2667,  0.2571]])

    """

    def __init__(self, lower=0., upper=1.):
        super(Uniform, self).__init__(
            dist=uniform(),
            lower=lower,
            upper=upper,
        )
