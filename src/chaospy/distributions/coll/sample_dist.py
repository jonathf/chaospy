"""A distribution that is based on a kernel density estimator (KDE)."""
import numpy
from scipy import special
from scipy.stats import gaussian_kde

from ..baseclass import Dist
from ..operators.addition import Add



class sample_dist(Dist):
    """A distribution that is based on a kernel density estimator (KDE)."""
    def __init__(self, kernel, lo, up):
        self.kernel = kernel
        super(sample_dist, self).__init__(lo=lo, up=up)

    def _cdf(self, x, lo, up):
        cdf_vals = np.zeros(x.shape)
        for i in range(0, len(x)):
            cdf_vals[i] = [self.kernel.integrate_box_1d(0, x_i) for x_i in x[i]]
        return cdf_vals

    def _pdf(self, x, lo, up):
        return self.kernel(x)

    def _bnd(self, lo, up):
        return (lo, up)

    def sample(self, size=(), rule="R", antithetic=None,
            verbose=False, **kws):
        """
            Overwrite sample() function, because the constructed Dist that is
            based on the KDE is only working with the random sampling that is
            given by the KDE itself.
        """

        size_ = np.prod(size, dtype=int)
        dim = len(self)
        if dim>1:
            if isinstance(size, (tuple,list,np.ndarray)):
                shape = (dim,) + tuple(size)
            else:
                shape = (dim, size)
        else:
            shape = size

        out = self.kernel.resample(size_)[0]
        try:
            out = out.reshape(shape)
        except:
            if len(self)==1:
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
        samples:
            Sample values to construction of the KDE
        lo (float) : Location of lower threshold
        up (float) : Location of upper threshold
    """
    if lo is None:
        lo = samples.min()
    if up is None:
        up = samples.max()

    try:
        #construct the kernel density estimator
        kernel = gaussian_kde(samples, bw_method="scott")
        dist = sample_dist(kernel, lo, up)
        dist.addattr(str="SampleDist(%s,%s)" % (lo, up))

    #raised by gaussian_kde if dataset is singular matrix
    except numpy.linalg.LinAlgError:
        dist = Uniform(lo=-numpy.inf, up=numpy.inf)

    return dist
