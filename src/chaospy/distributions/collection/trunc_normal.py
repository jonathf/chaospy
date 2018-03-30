"""Truncated normal distribution."""
import numpy
from scipy import special

from ..baseclass import Dist


class TruncNormal(Dist):
    """
    Truncated normal distribution

    Args:
        lower (float, Dist): Location of lower threshold
        upper (float, Dist): Location of upper threshold
        mu (float, Dist): Mean of normal distribution
        sigma (float, Dist): Standard deviation of normal distribution

    Examples:
        >>> distribution = chaospy.TruncNormal(2, 4, 2, 2)
        >>> print(distribution)
        TruncNormal(lower=2, mu=2, sigma=2, upper=4)
        >>> q = numpy.linspace(0, 1, 5)
        >>> print(numpy.around(distribution.inv(q), 4))
        [2.     2.4311 2.8835 3.387  4.    ]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.   0.25 0.5  0.75 1.  ]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.1687 1.1419 1.0601 0.9189 0.7089]
        >>> print(numpy.around(distribution.sample(4), 4))
        [3.1841 2.1971 3.8643 2.8501]
    """

    def __init__(self, lower=-1, upper=1, mu=0, sigma=1):
        self._repr = {"lower": lower, "upper": upper, "mu": mu, "sigma": sigma}
        Dist.__init__(self, a=lower, b=upper, sigma=sigma, mu=mu)

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
        return (x - fa) / (fb-fa)

    def _ppf(self, q, a, b, mu, sigma):
        fa = special.ndtr((a-mu)/sigma)
        fb = special.ndtr((b-mu)/sigma)
        return special.ndtri(q*(fb-fa) + fa)*sigma + mu

    def _bnd(self, a, b, mu, sigma):
        return a, b
