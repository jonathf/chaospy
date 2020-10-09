"""Truncated normal distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution
from ..operators import J



class trunc_normal(SimpleDistribution):

    def __init__(self, lower=-1, upper=1, mu=0, sigma=1):
        super(trunc_normal, self).__init__(
            parameters=dict(a=lower, b=upper, mu=mu, sigma=sigma),
            repr_args=["lower=%s" % lower, "upper=%s" % upper,
                       "mu=%s" % mu, "sigma=%s" % sigma],
        )

    def _pdf(self, x, a, b, mu, sigma):
        fa = special.ndtr((a-mu)/sigma)
        fb = special.ndtr((b-mu)/sigma)
        x = (x-mu)/sigma
        norm = (2*numpy.pi)**(-.5)*numpy.e**(-x**2/2.)
        return norm/(fb-fa)

    def _cdf(self, x, a, b, mu, sigma):
        fa = special.ndtr((a-mu)/sigma)
        fb = special.ndtr((b-mu)/sigma)
        x = special.ndtr((x-mu)/sigma)
        return (x-fa)/(fb-fa)

    def _ppf(self, q, a, b, mu, sigma):
        fa = special.ndtr((a-mu)/sigma)
        fb = special.ndtr((b-mu)/sigma)
        return special.ndtri(q*(fb-fa) + fa)*sigma + mu

    def _lower(self, a, b, mu, sigma):
        return a

    def _upper(self, a, b, mu, sigma):
        return b


class TruncNormal(J):
    """
    Truncated normal distribution

    Args:
        lower (float, Distribution):
            Location of lower threshold
        upper (float, Distribution):
            Location of upper threshold
        mu (float, Distribution):
            Mean of normal distribution
        sigma (float, Distribution):
            Standard deviation of normal distribution

    Examples:
        >>> distribution = chaospy.TruncNormal(2, 4, 2, 2)
        >>> distribution
        TruncNormal(lower=2, upper=4, mu=2, sigma=2)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([2.    , 2.4311, 2.8835, 3.387 , 4.    ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([1.1687, 1.1419, 1.0601, 0.9189, 0.7089])
        >>> distribution.sample(4).round(4)
        array([3.1841, 2.1971, 3.8643, 2.8501])

    """

    def __init__(self, lower=-1, upper=1, mu=0, sigma=1):
        super(TruncNormal, self).__init__(
            trunc_normal(lower=lower, upper=upper, mu=mu, sigma=sigma))
        self._repr_args=["lower=%s" % lower, "upper=%s" % upper ,
                         "mu=%s" % mu, "sigma=%s" % sigma]
