"""Gaussian kernel density estimation."""
import numpy
from scipy import special

from .baseclass import KernelDensityBaseclass
from ..collection.mv_normal import MvNormal


class GaussianKDE(KernelDensityBaseclass):
    """
    Examples:
        >>> samples = [-1, 0, 1]
        >>> dist = chaospy.GaussianKDE(samples, 0.4**2)
        >>> dist.pdf([-1, -0.5, 0, 0.5, 1]).round(4)
        array([0.3471, 0.3047, 0.3617, 0.3047, 0.3471])
        >>> dist.cdf([-1, -0.5, 0, 0.5, 1]).round(4)
        array([0.1687, 0.3334, 0.5   , 0.6666, 0.8313])
        >>> dist.inv([0, 0.25, 0.5, 0.75, 1]).round(4)
        array([-3.7687, -0.7645,  0.    ,  0.7645,  3.9424])
        >>> dist.mom([1, 2, 3]).round(4)
        array([0.    , 0.8267, 0.    ])
        >>> # Does dist normalize to one
        >>> t = numpy.linspace(-4, 4, 1000000)
        >>> abs(numpy.mean(dist.pdf(t))*(t[-1]-t[0]) - 1)  # err
        1.0000000212340154e-06

        >>> samples = [[-1, 0, 1], [0, 1, 2]]
        >>> dist = chaospy.GaussianKDE(samples, 0.4)
        >>> dist.lower.round(4)
        array([-5.7434, -4.7434])
        >>> dist.upper.round(4)
        array([5.7434, 6.7434])
        >>> dist.pdf([[0, 0, 1, 1], [0, 1, 0, 1]]).round(4)
        array([0.0482, 0.0977, 0.008 , 0.0482])
        >>> dist.fwd([[0, 0, 1, 1], [0, 1, 0, 1]]).round(4)
        array([[0.5   , 0.5   , 0.8141, 0.8141],
               [0.1274, 0.5   , 0.0158, 0.1597]])
        >>> dist.inv([[0, 0, 1, 1], [0, 1, 0, 1]]).round(4)
        array([[-5.7379, -5.7379,  5.393 ,  5.393 ],
               [-4.7434,  5.0072, -4.7434,  6.5641]])
        >>> dist.mom([(0, 1, 1), (1, 0, 1)]).round(4)
        array([1.    , 0.    , 0.6667])

    """

    @staticmethod
    def _kernel(z_loc):
        """The underlying density kernel."""
        return numpy.prod(numpy.e**(-z_loc**2/2.)/numpy.sqrt(2*numpy.pi), axis=-1)

    def _ikernel(self, z_loc):
        """The integrand of the underlying density kernel."""
        kernel = 1
        if z_loc.shape[-1] > 1:
            kernel = self._kernel(z_loc[:, :, :-1])
        return kernel*special.ndtr(z_loc[:, :, -1])

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
        return (self.samples[idx]-7.5*numpy.sqrt(self.covariance[:, idx, idx]).T).min(-1)

    def _upper(self, idx, dim, cache):
        """Upper bounds."""
        return (self.samples[idx]+7.5*numpy.sqrt(self.covariance[:, idx, idx]).T).max(-1)
