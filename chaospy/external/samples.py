"""
In some cases a constructed distribution that are first and foremost data
driven. In such scenarios it make sense to make use of
`kernel density estimation`_ (KDE). In ``chaospy`` KDE can be accessed through
the :func:`SampleDist` constructor.

Basic usage of the :func:`SampleDist` constructor involves just passing the
data as input argument::

    >>> data = [3, 4, 5, 5]
    >>> distribution = chaospy.SampleDist(data)

This distribution can be used as any other distributions::

    >>> distribution.cdf([3, 3.5, 4, 4.5, 5]).round(4)
    array([0.    , 0.1932, 0.4279, 0.7043, 1.    ])
    >>> distribution.mom(1).round(4)
    4.25
    >>> distribution.sample(4).round(4)
    array([4.4131, 3.3111, 4.9139, 4.1042])

It also supports lower and upper bounds defining where the range is expected to
appear, which gives a slightly different distribution::

    >>> distribution = chaospy.SampleDist(data, lo=2, up=6)
    >>> distribution.cdf([3, 3.5, 4, 4.5, 5]).round(4)
    array([0.1344, 0.2543, 0.4001, 0.5716, 0.7552])

In addition multivariate distributions supported::

    >>> data = [[1, 2, 2, 3], [5, 5, 4, 3]]
    >>> distribution = chaospy.SampleDist(data)
    >>> distribution.sample(4).round(4)
    array([[2.3601, 1.7626, 1.094 , 2.6507],
           [3.9457, 4.5802, 4.9081, 3.9278]])

.. _kernel density estimation: \
https://en.wikipedia.org/wiki/Kernel_density_estimation
"""
import numpy
from scipy.stats import gaussian_kde
import chaospy

from chaospy.distributions import SimpleDistribution


class sample_dist(SimpleDistribution):
    """A distribution that is based on a kernel density estimator (KDE)."""

    def __init__(self, samples, lo, up):
        samples = numpy.asarray(samples)
        self.samples = samples
        self.kernel = gaussian_kde(samples, bw_method="scott")
        self.flo = self.kernel.integrate_box_1d(0, lo)
        self.fup = self.kernel.integrate_box_1d(0, up)
        self.unbound = numpy.all(lo == samples.min())
        self.unbound &= numpy.all(up == samples.max())
        super(sample_dist, self).__init__(
            parameters=dict(lo=lo, up=up),
            repr_args=[repr(samples), lo, up],
        )

    def _cdf(self, xloc, lo, up):
        cdf_vals = numpy.array([self.kernel.integrate_box_1d(0, x)
                                for x in xloc])
        return (cdf_vals-self.flo)/(self.fup-self.flo)

    def _pdf(self, x, lo, up):
        return self.kernel(x)

    def _lower(self, lo, up):
        return lo

    def _upper(self, lo, up):
        return up

    def _mom(self, k, lo, up):
        if self.unbound:
            return numpy.prod(numpy.mean(self.samples.T**k, -1))
        raise chaospy.StochasticallyDependentError("component lack support")


def SampleDist(samples, lo=None, up=None, threshold=1e-5):
    """
    Distribution based on samples.

    Estimates a distribution from the given samples by constructing a kernel
    density estimator (KDE).

    Args:
        samples (numpy.ndarray):
            Sample values to construction of the KDE. Either shape
            ``(N,)`` or ``(D, N)``, where ``N`` are the number of
            samples, and ``D`` is the number of dimension in the
            distribution.
        lo (float):
            Location of lower bound.
        up (float):
            Location of upper bound.
        threshold (float):
            Threshold for how low the correlation between two
            columns should be before defining them as
            stochastically independent.

    Example:
        >>> distribution = chaospy.SampleDist([0, 1, 1, 1, 2])
        >>> distribution
        sample_dist(array([0, 1, 1, 1, 2]), 0, 2)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([0.    , 0.6016, 1.    , 1.3984, 2.    ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.2254, 0.4272, 0.5135, 0.4272, 0.2254])
        >>> distribution.sample(4).round(4)
        array([1.0294, 1.0993, 0.9116, 0.3768])
        >>> distribution.mom(1).round(4)
        1.0

    """
    samples = numpy.atleast_2d(samples)
    assert samples.ndim == 2, "samples have too many dimensions provided"

    if lo is None:
        lo = samples.min(axis=-1)
    else:
        lo = numpy.broadcast_to(lo, len(samples))
    if up is None:
        up = samples.max(axis=-1)
    else:
        up = numpy.broadcast_to(up, len(samples))

    # construct vector of marginals
    distributions = []
    for samples_, lo_, up_ in zip(samples, lo, up):
        #construct the kernel density estimator
        try:
            dist = sample_dist(samples_, lo_, up_)
        #raised by gaussian_kde if dataset is singular matrix
        except numpy.linalg.LinAlgError:
            dist = chaospy.Uniform(lower=-numpy.inf, upper=numpy.inf)
        distributions.append(dist)

    if len(samples) == 1:
        distributions = distributions[0]

    else:
        distributions = chaospy.J(*distributions)

        # Attach dependencies to data.
        correlation = numpy.corrcoef(samples)
        correlation[numpy.abs(correlation) <= threshold] = 0
        if numpy.any(correlation != numpy.diag(numpy.diag(correlation))):
            distributions = chaospy.Nataf(distributions, correlation)

    return distributions
