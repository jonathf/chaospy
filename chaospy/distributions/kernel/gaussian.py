"""Gaussian kernel density estimation."""
import numpy
from scipy import special

from .baseclass import KernelDensityBaseclass
from ..collection.mv_normal import MvNormal


class GaussianKDE(KernelDensityBaseclass):
    """
    Gaussian kernel density estimator.

    Density estimator that handles both univariate and multivariate data. It
    provides automatic bandwidth selection method using Scott's and Silverman's
    method.

    Attributes:
        samples:
            The raw data as provided by the user reshaped to have `ndim == 2`.
        covariance:
            The covariance matrix to each sample. Assuming uncorrelated
            dimensions, the bandwidth is the square root of the diagonals. It
            will have either dimensions `(1, n_dim, n_dim)` if all samples
            shares covariance, or `(n_samples, n_dim, n_dim)` if not.
        weights:
            How much each sample is weighted. Either a scalar when the samples
            are equally weighted, or with length `n_samples` otherwise.

    Examples:
        >>> samples = [[-1, 0, 1], [0, 1, 2]]
        >>> dist = chaospy.GaussianKDE(samples, 0.4)
        >>> uloc = [[0, 0, 1, 1], [0, 1, 0, 1]]
        >>> dist.pdf(uloc).round(4)
        array([0.0482, 0.0977, 0.008 , 0.0482])
        >>> dist.fwd(uloc).round(4)
        array([[0.5   , 0.5   , 0.8141, 0.8141],
               [0.1274, 0.5   , 0.0158, 0.1597]])
        >>> dist.inv(uloc).round(4)
        array([[-6.6758, -6.6758,  5.4719,  5.4719],
               [-4.454 ,  4.6876, -6.0671,  7.8807]])
        >>> dist.mom([(0, 1, 1), (1, 0, 1)]).round(4)
        array([1.    , 0.    , 0.6667])

    """

    @staticmethod
    def _kernel(z_loc):
        """The underlying density kernel."""
        return numpy.prod(numpy.e**(-z_loc**2/2.)/numpy.sqrt(2*numpy.pi), axis=-1)

    @staticmethod
    def _ikernel(z_loc):
        """The integrand of the underlying density kernel."""
        return special.ndtr(z_loc)

    def _mom(self, k_loc, cache):
        """Raw statistical moments."""
        length = self.samples.shape[-1]
        covariance = numpy.broadcast_to(
            self.covariance, (length,)+self.covariance.shape[1:])
        out = numpy.array([
            MvNormal._mom(
                self,
                k_loc,
                mean=self.samples[:, idx],
                sigma=covariance[idx],
                cache={},
            )
            for idx in range(length)
        ])
        return numpy.sum(out*self.weights)

    def _lower(self, idx, dim, cache):
        """Lower bounds."""
        return (self.samples[idx]-10*numpy.sqrt(self.covariance[:, idx, idx]).T).min(-1)

    def _upper(self, idx, dim, cache):
        """Upper bounds."""
        return (self.samples[idx]+10*numpy.sqrt(self.covariance[:, idx, idx]).T).max(-1)
