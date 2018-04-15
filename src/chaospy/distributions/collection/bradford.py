"""Bradford distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add


class bradford(Dist):
    """Standard Bradford distribution."""

    def __init__(self, c=1):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        return  c / (c*x + 1.0) / numpy.log(1.0+c)

    def _cdf(self, x, c):
        return numpy.log(1.0+c*x) / numpy.log(c+1.0)

    def _ppf(self, q, c):
        return ((1.0+c)**q-1)/c

    def _bnd(self, x, c):
        return 0, 1


class Bradford(Add):
    """
    Bradford distribution.

    Args:
        shape (float, Dist) : Shape parameter
        lower (float, Dist) : Location of lower threshold
        upper (float, Dist) : Location of upper threshold

    Examples:
        >>> distribution = chaospy.Bradford(0.8, 4, 6)
        >>> print(distribution)
        Bradford(lower=4, shape=0.8, upper=6)
        >>> q = numpy.linspace(0, 1, 5)
        >>> print(numpy.around(distribution.inv(q), 4))
        [4.     4.3957 4.8541 5.385  6.    ]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.   0.25 0.5  0.75 1.  ]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.6805 0.5875 0.5072 0.4379 0.3781]
        >>> print(numpy.around(distribution.sample(4), 4))
        [5.171  4.1748 5.8704 4.8192]
        >>> print(numpy.around(distribution.mom(1), 4))
        5.0
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[5.0195 5.0028 5.0009]
         [0.3314 0.2664 0.2571]]
    """
    def __init__(self, shape=1, lower=0, upper=1):
        self._repr = {"shape": shape, "lower": lower, "upper": upper}
        Add.__init__(self, left=bradford(shape)*(upper-lower), right=lower)
