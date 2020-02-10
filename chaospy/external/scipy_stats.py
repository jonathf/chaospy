"""
Historically, the syntax for ``chaospy``'s distributions is inspired by the
``scipy.stats`` distributions using the syntax. Over time though, they have
developed into different entities making them incompatible. The
:class:`ScipyStatsDist` exists to restore this incompatibility again.

Basic usage of the :class:`ScipyStatsDist` constructor involves just passing
the ``scipy.stats`` distributions as input argument::

    >>> from scipy.stats import norm
    >>> st_distribution = norm(0, 1)
    >>> distribution = chaospy.ScipyStatsDist(st_distribution)
    >>> distribution
    ScipyStatsDist(norm(0, 1))

This distribution then behaves as a normal ``chaospy`` distribution::

    >>> distribution.pdf([-1, 0, 1]).round(4)
    array([0.242 , 0.3989, 0.242 ])
    >>> distribution.mom([0, 1, 2])
    array([1., 0., 1.])

Currently the wrapper is limited to only support univariate distributions.
"""
import numpy

from ..distributions import Dist


class ScipyStatsDist(Dist):
    """
    One dimensional ``scipy.stats`` distribution.

    Args:
        distribution (openturns.Distribution):
            1D distribution created with ``scipy.stats``.
    """

    def __init__(self, distribution):
        from scipy.stats._distn_infrastructure import rv_frozen
        assert isinstance(distribution, rv_frozen), (
            "Expected frozen distribution from ``scipy.stats``.")
        assert not hasattr(distribution, "dim") or distribution.dim == 1, (
            "Only one-dimensional ``scipy.stats`` models supported.")
        Dist.__init__(self)
        self.distribution = distribution

    def _pdf(self, x_loc):
        return self.distribution.pdf(x_loc)

    def _cdf(self, x_loc):
        return self.distribution.cdf(x_loc)

    def _ppf(self, q_loc):
        return self.distribution.ppf(q_loc)

    def _lower(self):
        return self.distribution.interval(1-1e-14)[0]

    def _upper(self):
        return self.distribution.interval(1-1e-14)[1]

    def _mom(self, k_loc):
        return self.distribution.moment(int(k_loc))

    def __str__(self):
        name = self.distribution.dist.__class__.__name__
        name = name.replace("_gen", "")
        args = ", ".join([str(arg) for arg in self.distribution.args])
        return "%s(%s(%s))" % (self.__class__.__name__, name, args)
