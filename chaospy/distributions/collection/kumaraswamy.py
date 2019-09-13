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

    def _bnd(self, x, a, b):
        return 0,1


class Kumaraswamy(Add):
    """
    Kumaraswamy's double bounded distribution

    Args:
        alpha (float, Dist): First shape parameter, alpha > 0
        beta (float, Dist): Second shape parameter, b > 0
        lower (float, Dist): Lower threshold
        upper (float, Dist): Upper threshold

    Examples:
        >>> distribution = chaospy.Kumaraswamy(2, 2, 2, 3)
        >>> print(distribution)
        Kumaraswamy(alpha=2, beta=2, lower=2, upper=3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [2.3249 2.4748 2.6063 2.7435]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.1625 1.471  1.5337 1.33  ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [2.6414 2.2434 2.8815 2.5295]
        >>> print(numpy.around(distribution.mom(1), 4))
        2.5333
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[2.5056 2.5018 2.5008]
         [0.0489 0.0569 0.0595]]
    """

    def __init__(self, alpha, beta, lower=0, upper=1):
        self._repr = {
            "alpha": alpha, "beta": beta, "lower": lower, "upper": upper}
        Add.__init__(
            self, left=kumaraswamy(alpha, beta)*(upper-lower), right=lower)
