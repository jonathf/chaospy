"""Kernel density estimation baseclass."""
from __future__ import division

import numpy
import chaospy

from ..baseclass import Distribution


class KernelDensityBaseclass(Distribution):
    """Kernel density estimation baseclass."""

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
        samples = numpy.atleast_2d(samples)
        assert samples.ndim == 2

        # the scale is taken from Scott-92.
        # The Scott factor is taken from scipy docs.
        if h_mat is None:

            if estimator_rule in ("scott", "silverman"):
                qrange = numpy.quantile(samples, [0.25, 0.75], axis=1).ptp(axis=0)
                scale = numpy.min([numpy.std(samples, axis=1), qrange/1.34], axis=0)
                factor = samples.shape[1]
                if estimator_rule == "silverman":
                    factor *= (len(samples)+2)/4.
                factor **= -1./(len(samples)+4)
                covariance = numpy.diag(scale*factor)**2

            else:
                raise ValueError("unknown estimator rule: %s" % estimator_rule)

        else:
            covariance = numpy.asfarray(h_mat)
            if covariance.ndim in (0, 1):
                covariance = covariance*numpy.eye(len(samples))
        if covariance.ndim == 2:
            covariance = covariance[numpy.newaxis]
        else:
            covariance = numpy.rollaxis(covariance, 2, 0)
        assert covariance.shape[1:] == (len(samples), len(samples))

        if weights is None:
            weights = 1./samples.shape[1]
        self.weights = weights

        dependencies, _, rotation = chaospy.declare_dependencies(
            self, dict(), rotation=rotation,
            dependency_type="accumulate", length=len(samples),
        )

        self._samples = samples
        self._covariance = covariance
        self._permute = numpy.eye(len(rotation), dtype=int)[rotation]
        self._pcovariance = numpy.matmul(numpy.matmul(
            self._permute, covariance), self._permute.T)
        cholesky = numpy.linalg.cholesky(self._pcovariance)
        self._fwd_transform = numpy.linalg.inv(cholesky)
        self._inv_transform = cholesky

        super(KernelDensityBaseclass, self).__init__(
            parameters={},
            dependencies=dependencies,
            rotation=rotation,
        )
        self._zloc = None
        self._kernel0 = None
        self._kernel1 = None

    def get_parameters(self, idx, cache, assert_numerical=True):
        parameters = super(KernelDensityBaseclass, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        if idx is None:
            del parameters["idx"]
        else:
            parameters["dim"] = dim = self._rotation.index(idx)
        return parameters

    def _pdf(self, x_loc, idx, dim, cache):
        """Kernel density function."""
        s, t = numpy.mgrid[:x_loc.shape[-1], :self._samples.shape[-1]]
        if not dim:
            samples = self._samples[idx, t]
            z_loc = ((x_loc[s]-samples)*self._fwd_transform[:, 0, 0])
            self._zloc = z_loc[:, :, numpy.newaxis]
            kernel = self._kernel(self._zloc)/self._inv_transform[:, 0, 0]
            self._kernel0 = self._kernel1 = kernel
            out = numpy.sum(kernel*self.weights, axis=-1)

        else:
            if self._zloc.shape[2] == dim+1:
                self._kernel0 = self._kernel1
            x_loc = [self._get_cache(dim_, cache, get=0)
                     for dim_ in self._rotation[:dim]] + [x_loc]
            x_loc = numpy.dstack([x[s] for x in x_loc])
            samples = numpy.dstack([self._samples[dim_, t]
                                    for dim_ in self._rotation[:dim+1]])
            zloc = numpy.sum((x_loc-samples)*self._fwd_transform[:, dim, :dim+1], -1)
            self._zloc = numpy.dstack([self._zloc[:, :, :dim], zloc])

            kernel = self._kernel(self._zloc)
            kernel *= (numpy.linalg.det(self._inv_transform[:, :dim, :dim])/
                       numpy.linalg.det(self._inv_transform[:, :dim+1, :dim+1]))
            out = (numpy.sum(kernel*self.weights, axis=-1)/
                   numpy.sum(self._kernel0*self.weights, axis=-1))
            self._kernel1, self._kernel0 = self._kernel0, kernel

        return out

    def _cdf(self, x_loc, idx, dim, cache):
        """Forward mapping."""
        s, t = numpy.mgrid[:x_loc.shape[-1], :self._samples.shape[-1]]
        if not dim:
            z_loc = (x_loc[s]-self._samples[idx, t])*self._fwd_transform[:, 0, 0]
            self._zloc = z_loc[:, :, numpy.newaxis]
            out = numpy.sum(self._ikernel(z_loc)*self.weights, axis=-1)
            assert out.shape == x_loc.shape, (out.shape, x_loc.shape)

        else:
            x_loc = [self._get_cache(dim_, cache, get=0)
                     for dim_ in self._rotation[:dim]] + [x_loc]
            x_loc = numpy.dstack([x[s] for x in x_loc])

            samples = numpy.dstack([self._samples[dim_, t]
                                    for dim_ in self._rotation[:dim+1]])
            zloc = numpy.sum((x_loc-samples)*self._fwd_transform[:, dim, :dim+1], -1)
            self._zloc = numpy.dstack([self._zloc[:, :, :dim], zloc])
            ikernel = self._kernel(self._zloc[:, :, :-1])*self._ikernel(self._zloc[:, :, -1])
            out = (numpy.sum(ikernel*self.weights, axis=-1)/
                   numpy.sum(self._kernel(self._zloc[:, :, :-1])*self.weights, axis=-1))
        return out

    def _ppf(self, u_loc, idx, dim, cache):
        """Inverse mapping."""
        xloc0 = numpy.quantile(self._samples[idx], u_loc)
        out = chaospy.approximate_inverse(
            self, idx, u_loc, xloc0=xloc0, cache=cache, iterations=1000)
        return out
