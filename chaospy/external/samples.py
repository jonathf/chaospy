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

    >>> print(numpy.around(distribution.cdf([3, 3.5, 4, 4.5, 5]), 4))
    [0.     0.1932 0.4279 0.7043 1.    ]
    >>> print(numpy.around(distribution.mom(1), 4))
    4.0922

It also supports lower and upper bounds defining where the range is expected to
appear, which gives a slightly different distribution::

    >>> distribution = chaospy.SampleDist(data, lo=2, up=6)
    >>> print(numpy.around(distribution.cdf([3, 3.5, 4, 4.5, 5]), 4))
    [0.1344 0.2543 0.4001 0.5716 0.7552]
    >>> print(numpy.around(distribution.mom(1), 4))
    4.2149

Currently the wrapper is limited to only support univariate distributions.

.. _kernel density estimation: \
https://en.wikipedia.org/wiki/Kernel_density_estimation
"""
import numpy
from scipy.stats import gaussian_kde

from chaospy.distributions.baseclass import Dist
from chaospy.distributions.operators.addition import Add
from chaospy.distributions.collection.uniform import Uniform
from chaospy.distributions.collection.deprecate import deprecation_warning


class sample_dist(Dist):
    """A distribution that is based on a kernel density estimator (KDE)."""
    def __init__(self, samples, lo, up):
        self.samples = samples
        self.kernel = gaussian_kde(samples, bw_method="scott")
        self.flo = self.kernel.integrate_box_1d(0, lo)
        self.fup = self.kernel.integrate_box_1d(0, up)
        super(sample_dist, self).__init__(lo=lo, up=up)

    def _cdf(self, x, lo, up):
        cdf_vals = numpy.zeros(x.shape)
        for i in range(0, len(x)):
            cdf_vals[i] = [self.kernel.integrate_box_1d(0, x_i) for x_i in x[i]]
        cdf_vals = (cdf_vals - self.flo) / (self.fup - self.flo)
        return cdf_vals

    def _pdf(self, x, lo, up):
        return self.kernel(x)

    def _lower(self, lo, up):
        return lo

    def _upper(self, lo, up):
        return up

    def sample(self, size=(), rule="R", antithetic=None, verbose=False, **kws):
        """
        Overwrite sample() function, because the constructed Dist that is
        based on the KDE is only working with the random sampling that is
        given by the KDE itself.
        """
        size_ = numpy.prod(size, dtype=int)
        dim = len(self)
        if dim > 1:
            if isinstance(size, (tuple,list,numpy.ndarray)):
                shape = (dim,) + tuple(size)
            else:
                shape = (dim, size)
        else:
            shape = size

        out = self.kernel.resample(size_)[0]
        try:
            out = out.reshape(shape)
        except:
            if len(self) == 1:
                out = out.flatten()
            else:
                out = out.reshape(dim, out.size/dim)

        return out


def SampleDist(samples, lo=None, up=None):
    """
    Distribution based on samples.

    Estimates a distribution from the given samples by constructing a kernel
    density estimator (KDE).

    Args:
        samples (numpy.ndarray):
            Sample values to construction of the KDE
        lo (float):
            Location of lower threshold
        up (float):
            Location of upper threshold

    Example:
        >>> distribution = chaospy.SampleDist([0, 1, 1, 1, 2])
        >>> distribution
        sample_dist(lo=0, up=2)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([0.    , 0.6016, 1.    , 1.3984, 2.    ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.2254, 0.4272, 0.5135, 0.4272, 0.2254])
        >>> distribution.sample(4).round(4)
        array([ 1.5877,  1.1645, -0.0131,  1.3302])
        >>> distribution.mom(1).round(4)
        1.0
    """
    samples = numpy.asarray(samples)
    if lo is None:
        lo = samples.min()
    if up is None:
        up = samples.max()

    try:
        #construct the kernel density estimator
        dist = sample_dist(samples, lo, up)

    #raised by gaussian_kde if dataset is singular matrix
    except numpy.linalg.LinAlgError:
        dist = Uniform(lower=-numpy.inf, upper=numpy.inf)

    return dist
