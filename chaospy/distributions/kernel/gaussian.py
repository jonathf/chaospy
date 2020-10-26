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
        h_mat:
            The covariance matrix to each sample. Assuming uncorrelated
            dimensions, the bandwidth is the square root of the diagonals. It
            will have either dimensions `(1, n_dim, n_dim)` if all samples
            shares covariance, or `(n_samples, n_dim, n_dim)` if not.
        weights:
            How much each sample is weighted. Either a scalar when the samples
            are equally weighted, or with length `n_samples` otherwise.

    Examples:
        >>> samples = [[-1, 0, 1], [0, 1, 2]]
        >>> distribution = chaospy.GaussianKDE(samples, estimator_rule="silverman")
        >>> distribution.h_mat  # H-matrix or bandwidth**2
        array([[[0.38614462, 0.        ],
                [0.        , 0.38614462]]])
        >>> uloc = [[0, 0, 1, 1], [0, 1, 0, 1]]
        >>> distribution.pdf(uloc).round(4)
        array([0.0469, 0.0982, 0.0074, 0.0469])
        >>> distribution.fwd(uloc).round(4)
        array([[0.5   , 0.5   , 0.8152, 0.8152],
               [0.1233, 0.5   , 0.0142, 0.1532]])
        >>> distribution.inv(uloc).round(4)
        array([[-6.577 , -6.577 ,  5.3948,  5.3948],
               [-4.3871,  4.6411, -5.9611,  7.7779]])
        >>> distribution.mom([(0, 1, 1), (1, 0, 1)]).round(4)
        array([1.    , 0.    , 0.6667])

    """

    @property
    def samples(self):
        return self._samples

    @property
    def h_mat(self):
        return self._covariance

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
        h_mat = numpy.broadcast_to(
            self.h_mat, (length,)+self.h_mat.shape[1:])
        out = numpy.array([
            MvNormal._mom(
                self,
                k_loc,
                mean=self.samples[:, idx],
                sigma=h_mat[idx],
                cache={},
            )
            for idx in range(length)
        ])
        return numpy.sum(out*self.weights)

    def _lower(self, idx, dim, cache):
        """Lower bounds."""
        del dim
        del cache
        return (self.samples[idx]-10*numpy.sqrt(self.h_mat[:, idx, idx]).T).min(-1)

    def _upper(self, idx, dim, cache):
        """Upper bounds."""
        del dim
        del cache
        return (self.samples[idx]+10*numpy.sqrt(self.h_mat[:, idx, idx]).T).max(-1)
