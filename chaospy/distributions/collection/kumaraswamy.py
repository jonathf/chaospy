"""Kumaraswswamy's double bounded distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class kumaraswamy(Dist):
    """Kumaraswswamy's double bounded distribution."""

    def __init__(self, a=1, b=1):
        assert numpy.all(a > 0) and numpy.all(b > 0)
        Dist.__init__(self, a=a, b=b)

    def _pdf(self, x, a, b):
        return a*b*x**(a-1)*(1-x**a)**(b-1)

    def _cdf(self, x, a, b):
        return 1-(1-x**a)**b

    def _ppf(self, q, a, b):
        return (1-(1-q)**(1./b))**(1./a)

    def _mom(self, k, a, b):
        return b*special.gamma(1+k*1./a)*special.gamma(b)/\
                special.gamma(1+b+k*1./a)

    def _lower(self, a, b):
        return 0.

    def _upper(self, a, b):
        return 1.


class Kumaraswamy(Add):
    """
    Kumaraswamy's double bounded distribution

    Args:
        alpha (float, Dist):
            First shape parameter, alpha > 0
        beta (float, Dist):
            Second shape parameter, b > 0
        lower (float, Dist):
            Lower threshold
        upper (float, Dist):
            Upper threshold

    Examples:
        >>> distribution = chaospy.Kumaraswamy(2, 2, 2, 3)
        >>> distribution
        Kumaraswamy(alpha=2, beta=2, lower=2, upper=3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.3249, 2.4748, 2.6063, 2.7435])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([1.1625, 1.471 , 1.5337, 1.33  ])
        >>> distribution.sample(4).round(4)
        array([2.6414, 2.2434, 2.8815, 2.5295])
        >>> distribution.mom(1).round(4)
        2.5333
    """

    def __init__(self, alpha, beta, lower=0, upper=1):
        self._repr = {
            "alpha": alpha, "beta": beta, "lower": lower, "upper": upper}
        Add.__init__(
            self, left=kumaraswamy(alpha, beta)*(upper-lower), right=lower)
