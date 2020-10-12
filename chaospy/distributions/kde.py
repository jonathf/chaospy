"""Gaussian kernel density estimation."""
from __future__ import division
from typing import Callable

import numpy
from scipy.special import comb, ndtr, ndtri, factorial2
from scipy.stats import gaussian_kde
import chaospy

from .baseclass import Distribution
from .collection.mv_normal import MvNormal


class GaussianKDE(Distribution):
    """
    Examples:
        >>> samples = [-1, 0, 1]
        >>> dist = GaussianKDE(samples, 0.4**2)
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
        >>> dist = GaussianKDE(samples, 0.4)
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

    def __init__(
            self,
            samples,
            h_mat=None,
            estimator_rule="scott",
            weights=None,
            rotation=None,
    ):
        """
        Args:
            samples (numpy.ndarray):
                The samples to generate density estimation. Assumed to have
                shape either as `(n_samples,)`, or `(n_dim, n_samples)`.
            h_mat (Optional[numpy.ndarray]):
                The H-matrix, also known as the smoothing matrix or bandwidth
                matrix. In one dimension it correspond to the square of the
                bandwidth parameters often used in the one-dimensional case.
                Assumes shape either something compatible with
                `(n_dim, n_dim)`, or `(n_dim, n_dim, n_samples)` in the case
                where each sample have their own H-matrix. If omitted, it is
                automatically calculated using `estimator_rule`.
            estimator_rule (str):
                Which method to use to select smoothing matrix from, assuming
                it is omitted. Choose from 'scott' and 'silverman'.
            weights (Optional[numpy.ndarray]):
                Weights of the samples. This must have the shape
                `(n_samples,)`. If omitted, each sample is assumed to be
                equally weighted.

        """
        self.samples = numpy.atleast_2d(samples)
        assert self.samples.ndim == 2

        # the scale is taken from Scott-92.
        # The Scott factor is taken from scipy docs.
        if h_mat is None:

            if estimator_rule == "scott":
                qrange = numpy.quantile(self.samples, [0.25, 0.75], axis=1).ptp(axis=0)
                scale = numpy.min([numpy.std(samples, axis=1), qrange/1.34], axis=0)
                factor = self.samples.shape[1]**(-1./(len(self.samples)+4))
                covariance = numpy.diag(scale*factor)**2

            elif estimator_rule == "silverman":
                qrange = numpy.quantile(self.samples, [0.25, 0.75], axis=1).ptp(axis=0)
                scale = numpy.min([numpy.std(samples, axis=1), qrange/1.34], axis=0)
                factor = (self.samples.shape[1]*(len(self.samples)+2)/4.)**(-1./(len(self.samples)+4))
                covariance = numpy.diag(scale*factor)**2

            else:
                raise ValueError("unknown estimator rule: %s" % estimator_rule)

        else:
            covariance = numpy.asfarray(h_mat)
            if covariance.ndim in (0, 1):
                covariance = covariance*numpy.eye(len(self.samples))
        if covariance.ndim == 2:
            covariance = covariance[numpy.newaxis]
        else:
            covariance = numpy.rollaxis(covariance, 2, 0)
        assert covariance.shape[1:] == (len(self.samples), len(self.samples))

        if weights is None:
            weights = 1./self.samples.shape[1]
        self.weights = weights

        self.covariance = covariance
        self.L = numpy.linalg.cholesky(covariance)
        self.Li = numpy.linalg.inv(self.L)

        dependencies, _, rotation = chaospy.declare_dependencies(
            self, dict(), rotation=rotation,
            dependency_type="accumulate", length=len(self.samples),
        )

        super(GaussianKDE, self).__init__(
            parameters={},
            dependencies=dependencies,
            rotation=rotation,
        )
        self._zloc = None
        self._samples = None
        self._kernel0 = None

    def get_parameters(self, idx, cache, assert_numerical=True):
        parameters = super(GaussianKDE, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        if idx is None:
            del parameters["idx"]
        else:
            parameters["dim"] = dim = self._rotation.index(idx)
        return parameters

    @staticmethod
    def _kernel(z_loc):
        """The underlying density kernel."""
        return numpy.prod(numpy.e**(-z_loc**2/2.)/numpy.sqrt(2*numpy.pi), axis=-1)

    def _pdf(self, x_loc, idx, dim, cache):
        """Kernel density function."""
        # grid up every location to evaluate against every sample
        s, t = numpy.mgrid[:x_loc.shape[-1], :self.samples.shape[-1]]


        # The first dimension
        if not dim:
            samples = self.samples[dim, t]
            z_loc = ((x_loc[s]-samples)*self.Li[:, 0, 0])[:, :, numpy.newaxis]
            self._zloc = z_loc
            self._kernel0 = self._kernel1 = self._kernel(z_loc)/self.L[:, 0, 0]
            out = numpy.sum(self._kernel0*self.weights, axis=-1)

        else:
            if self._zloc.shape[2] == dim+1:
                self._kernel0 = self._kernel1
            x_loc = [self._get_cache(dim_, cache, get=0)
                     for dim_ in self._rotation[:dim]] + [x_loc]
            x_loc = numpy.dstack([x[s] for x in x_loc])
            samples = numpy.dstack([self.samples[dim_, t]
                                    for dim_ in self._rotation[:dim+1]])
            zloc = numpy.sum((x_loc-samples)*self.Li[:, idx, :idx+1], -1)
            self._zloc = numpy.dstack([self._zloc[:, :, :dim], zloc])

            kernel = self._kernel(self._zloc)
            kernel *= (numpy.linalg.det(self.L[:, :idx, :idx])/
                       numpy.linalg.det(self.L[:, :idx+1, :idx+1]))
            out = (numpy.sum(kernel*self.weights, axis=-1)/
                   numpy.sum(self._kernel0*self.weights, axis=-1))
            self._kernel1, self._kernel0 = self._kernel0, kernel

        return out

    def _ikernel(self, z_loc):
        """The integrand of the underlying density kernel."""
        kernel = 1
        if z_loc.shape[-1] > 1:
            kernel = self._kernel(z_loc[:, :, :-1])
        return kernel*ndtr(z_loc[:, :, -1])

    def _cdf(self, x_loc, idx, dim, cache):
        """Forward mapping."""
        s, t = numpy.mgrid[:x_loc.shape[-1], :self.samples.shape[-1]]

        if not dim:
            self._zloc = ((x_loc[s]-self.samples[idx, t])*self.Li[:, 0, 0])[:, :, numpy.newaxis]
            out = numpy.sum(self._ikernel(self._zloc)*self.weights, axis=-1)

        else:
            x_loc = [self._get_cache(dim_, cache, get=0)
                     for dim_ in self._rotation[:dim]] + [x_loc]
            x_loc = numpy.dstack([x[s] for x in x_loc])

            samples = numpy.dstack([self.samples[dim_, t]
                                    for dim_ in self._rotation[:dim+1]])
            zloc = numpy.sum((x_loc-samples)*self.Li[:, idx, :idx+1], -1)
            self._zloc = numpy.dstack([self._zloc[:, :, :dim], zloc])
            out = (numpy.sum(self._ikernel(self._zloc)*self.weights, axis=-1)/
                   numpy.sum(self._kernel(self._zloc[:, :, :-1])*self.weights, axis=-1))


        return out

    def _ppf(self, u_loc, idx, dim, cache):
        """Inverse mapping."""
        # speed up convergence considerable, by giving very good initial position.
        xloc0 = None if dim else numpy.quantile(self.samples[idx], u_loc)
        return chaospy.approximate_inverse(
            self, idx, u_loc, xloc0=xloc0, cache=cache, tolerance=1e-12)

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
