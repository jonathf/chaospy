"""Laplace Probability Distribution."""
import numpy
from scipy import special
import chaospy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class laplace(SimpleDistribution):
    """Laplace Probability Distribution."""

    def __init__(self):
        super(laplace, self).__init__()

    def _pdf(self, x):
        return numpy.e**-numpy.abs(x)/2

    def _cdf(self, x):
        return (1+numpy.sign(x)*(1-numpy.e**-abs(x)))/2

    def _mom(self, k):
        return special.factorial(k)*((k+1)%2)

    def _ppf(self, x):
        return numpy.where(x>.5, -numpy.log(2*(1-x)), numpy.log(2*x))

    def _ttr(self, k):
        q1, w1 = chaospy.quad_fejer(500, (-25, 0))
        q2, w2 = chaospy.quad_fejer(500, (0, 25))
        q = numpy.concatenate([q1,q2], 1)
        w = numpy.concatenate([w1,w2])*self._pdf(q[0])

        coeffs, _, _ = chaospy.discretized_stieltjes(k, q, w)
        return coeffs[:, 0, -1]


class Laplace(ShiftScaleDistribution):
    R"""
    Laplace Probability Distribution

    Args:
        mu (float, Distribution):
            Mean of the distribution.
        scale (float, Distribution):
            Scaling parameter. scale > 0.

    Examples:
        >>> distribution = chaospy.Laplace(2, 2)
        >>> distribution
        Laplace(mu=2, sigma=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
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
               [ 8.    , 39.9996, 86.4   ]])

    """
    def __init__(self, mu=0, sigma=1):
        super(Laplace, self).__init__(
            dist=laplace(),
            scale=sigma,
            shift=mu,
        )
        self._repr_args = ["mu=%s" % mu, "sigma=%s" % sigma]
