"""Truncated normal distribution."""
import numpy
from scipy import special
from scipy.stats import truncnorm
import chaospy

from .normal import normal
from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class trunc_normal(SimpleDistribution):

    def __init__(self, lower=-1, upper=1, mu=0, sigma=1):
        super(trunc_normal, self).__init__(
            parameters=dict(a=lower, b=upper, mu=mu, sigma=sigma),
            repr_args=["lower=%s" % lower, "upper=%s" % upper,
                       "mu=%s" % mu, "sigma=%s" % sigma],
        )

    def _pdf(self, x, a, b, mu, sigma):
        return truncnorm.pdf(x, a, b, loc=mu, scale=sigma)

    def _cdf(self, x, a, b, mu, sigma):
        return truncnorm.cdf(x, a, b, loc=mu, scale=sigma)

    def _ppf(self, q, a, b, mu, sigma):
        return truncnorm.ppf(q, a, b, loc=mu, scale=sigma)

    def _lower(self, a, b, mu, sigma):
        del b
        lower = normal()._lower()*sigma+mu
        return numpy.where(a < lower, lower, a)

    def _upper(self, a, b, mu, sigma):
        del a
        upper = normal()._upper()*sigma+mu
        return numpy.where(b > upper, upper, b)

    def _mom(self, n, a, b, mu, sigma):
        return truncnorm.moment(int(n), a, b, loc=mu, scale=sigma)


class TruncNormal(ShiftScaleDistribution):
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
        >>> full_trunc = chaospy.TruncNormal(lower=-1, upper=1)
        >>> half_trunc = chaospy.TruncNormal(upper=1)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> half_trunc.inv(uloc).round(3)
        array([-8.22 , -0.961, -0.422,  0.012,  0.448,  1.   ])
        >>> xloc = full_trunc.inv(uloc)
        >>> xloc.round(3)
        array([-1.   , -0.538, -0.172,  0.172,  0.538,  1.   ])
        >>> numpy.allclose(full_trunc.fwd(xloc), uloc)
        True
        >>> full_trunc.pdf(xloc).round(3)
        array([0.354, 0.506, 0.576, 0.576, 0.506, 0.354])
        >>> half_trunc.pdf(xloc).round(3)
        array([0.288, 0.41 , 0.467, 0.467, 0.41 , 0.288])
        >>> full_trunc.sample(4).round(3)
        array([ 0.266, -0.715,  0.868, -0.03 ])
        >>> half_trunc.sample(4).round(3)
        array([ 0.625, -0.921, -1.822, -0.428])
        >>> full_trunc.mom([1, 2, 3])
        array([0.        , 0.29112509, 0.        ])
        >>> half_trunc.mom([1, 2, 3])
        array([-0.28759997,  0.71240003, -0.86279991])

    """

    def __init__(self, lower=-numpy.inf, upper=numpy.inf, mu=0, sigma=1):
        super(TruncNormal, self).__init__(
            trunc_normal(lower=lower, upper=upper, mu=mu, sigma=sigma))
        self._repr_args = chaospy.format_repr_kwargs(lower=(lower, -numpy.inf))
        self._repr_args += chaospy.format_repr_kwargs(upper=(upper, numpy.inf))
        self._repr_args += chaospy.format_repr_kwargs(mu=(mu, 0))
        self._repr_args += chaospy.format_repr_kwargs(sigma=(sigma, 1))
