"""Laplace Probability Distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add



class laplace(Dist):
    """Laplace Probability Distribution."""

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return numpy.e**-numpy.abs(x)/2

    def _cdf(self, x):
        return (1+numpy.sign(x)*(1-numpy.e**-abs(x)))/2

    def _mom(self, k):
        return .5*special.factorial(k)*(1+(-1)**k)

    def _ppf(self, x):
        return numpy.where(x>.5, -numpy.log(2*(1-x)), numpy.log(2*x))

    def _lower(self):
        return -32.

    def _upper(self):
        return 32.

    def _ttr(self, k):
        from ...quadrature import quad_fejer, discretized_stieltjes
        q1, w1 = quad_fejer(500, (-32, 0))
        q2, w2 = quad_fejer(500, (0, 32))
        q = numpy.concatenate([q1,q2], 1)
        w = numpy.concatenate([w1,w2])*self.pdf(q[0])

        coeffs, _, _ = discretized_stieltjes(k, q, w)
        return coeffs[:, 0, -1]


class Laplace(Add):
    R"""
    Laplace Probability Distribution

    Args:
        mu (float, Dist):
            Mean of the distribution.
        scale (float, Dist):
            Scaling parameter. scale > 0.

    Examples:
        >>> distribution = chaospy.Laplace(2, 2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([0.1674, 1.5537, 2.4463, 3.8326])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.1, 0.2, 0.2, 0.1])
        >>> distribution.sample(4).round(4)
        array([ 2.734 , -0.9392,  6.6165,  1.9275])
        >>> distribution.mom(1).round(4)
        2.0
        >>> distribution.ttr([1, 2, 3]).round(4)
        array([[ 2.    ,  2.    ,  2.    ],
               [ 8.    , 39.9995, 86.4011]])
    """
    def __init__(self, mu=0, scale=1):
        self._repr = {"mu": mu, "scale": scale}
        Add.__init__(self, left=laplace()*scale, right=mu)
