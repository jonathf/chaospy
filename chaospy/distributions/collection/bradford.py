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

    def _lower(self, c):
        return 0

    def _upper(self, c):
        return 1


class Bradford(Add):
    """
    Bradford distribution.

    Args:
        shape (float, Dist):
            Shape parameter
        lower (float, Dist):
            Location of lower threshold
        upper (float, Dist):
            Location of upper threshold

    Examples:
        >>> distribution = chaospy.Bradford(0.8, 4, 6)
        >>> distribution
        Bradford(lower=4, shape=0.8, upper=6)
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
        >>> distribution.ttr([1, 2, 3]).round(4)
        array([[5.0195, 5.0028, 5.0009],
               [0.3314, 0.2664, 0.2571]])
    """
    def __init__(self, shape=1, lower=0, upper=1):
        self._repr = {"shape": shape, "lower": lower, "upper": upper}
        Add.__init__(self, left=bradford(shape)*(upper-lower), right=lower)
