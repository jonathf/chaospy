"""Uniform probability distribution."""
from ..baseclass import Dist
from ..operators.addition import Add


class uniform(Dist):
    """Uniform distribution fixed on the [-1, 1] interval."""

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return 0.5

    def _cdf(self, x):
        return .5*x+.5

    def _ppf(self, q):
        return 2*q-1

    def _bnd(self, x):
        return -1.,1.

    def _mom(self, k):
        return 1./(k + 1)* (k % 2 == 0)

    def _ttr(self, n):
        return 0., n*n/(4.*n*n-1)


class Uniform(Add):
    r"""
    Uniform probability distribution.

    Args:
        lower (float, Dist): Lower threshold of distribution. Must be smaller
            than ``upper``.
        upper (float, Dist): Upper threshold of distribution.

    Examples:
        >>> distribution = chaospy.Uniform(2, 4)
        >>> print(distribution)
        Uniform(lower=2, upper=4)
        >>> q = numpy.linspace(0, 1, 5)
        >>> print(numpy.around(distribution.inv(q), 4))
        [2.  2.5 3.  3.5 4. ]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.   0.25 0.5  0.75 1.  ]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.5 0.5 0.5 0.5 0.5]
        >>> print(numpy.around(distribution.sample(4), 4))
        [3.3072 2.23   3.9006 2.9644]
        >>> print(distribution.mom(1))
        3.0
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[3.     3.     3.    ]
         [0.3333 0.2667 0.2571]]
    """

    def __init__(self, lower=0, upper=1):
        self._repr = {"lower": lower, "upper": upper}
        left = uniform()*((upper-lower)*.5)
        right = 0.5*(upper+lower)
        Add.__init__(self, left=left, right=right)
