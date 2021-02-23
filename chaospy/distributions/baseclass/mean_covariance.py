"""Mean-Covariance transformation."""
import numpy
from scipy.special import comb
import numpoly
import chaospy

from .distribution import Distribution


class MeanCovarianceDistribution(Distribution):
    """
    Mean-Covariance transformation.

    Transforms any distribution with zero mean and unit variance into provided
    `mean` and `covariance`. The transformation works on other distributions as
    well, but in those cases `mean` and `covariance` are likely not correct.

    Args:
        dist (Distribution):
            The underlying distribution to be scaled.
        mean (float, Sequence[float], Distribution):
            Mean vector.
        covariance (float, Sequence[Sequence[float]], Distribution):
            Covariance matrix or variance vector if scale is a 1-d vector.
            If omitted, assumed to be 1.
        rotation (Sequence[int], Sequence[Sequence[bool]]):
            The order of which to resolve conditionals. Either as a sequence of
            column rotations, or as a permutation matrix.
            Defaults to `range(len(distribution))` which is the same as
            `p(x0), p(x1|x0), p(x2|x0,x1), ...`.

    """

    def __init__(
            self,
            dist,
            mean=0,
            covariance=1,
            rotation=None,
            repr_args=None,
    ):
        assert isinstance(dist, Distribution), "'dist' should be a distribution"
        mean = mean if isinstance(mean, Distribution) else numpy.atleast_1d(mean)
        assert not isinstance(covariance, Distribution)
        length = max(len(dist), len(mean), len(covariance))

        if not isinstance(mean, Distribution):
            assert mean.ndim == 1, "Parameter 'mean' have too many dimensions"
            assert len(mean) == length
        assert len(mean) in (1, length)

        exclusion = dist._exclusion.copy()
        if len(dist) == 1 and length > 1:
            dist = chaospy.Iid(dist, length)

        covariance = numpy.asarray(covariance)
        assert covariance.ndim <= 2, (
            "Covariance must either be scalar, vector or matrix")
        if covariance.ndim == 0:
            covariance = numpy.eye(length)*covariance
        elif covariance.ndim == 1:
            covariance = numpy.diag(covariance)
        assert covariance.shape == (length, length), (
            "Parameters 'mean' and 'covariance' have shape mismatch.")
        assert len(covariance) == length

        if rotation is not None:
            rotation = list(rotation)
        else:
            rotation = [key for key, _ in sorted(enumerate(dist._dependencies),
                                                 key=lambda x: len(x[1]))]

        accumulant = set()
        dependencies = [deps.copy() for deps in dist._dependencies]
        for idx in rotation:
            accumulant.update(dist._dependencies[idx])
            dependencies[idx] = accumulant.copy()

        if isinstance(mean, Distribution):
            if len(mean) == 1:
                for dependency in dependencies:
                    dependency.update(mean._dependencies[0])
            else:
                for dep1, dep2 in zip(dependencies, mean._dependencies):
                    dep1.update(dep2)

        self._permute = numpy.eye(len(rotation), dtype=int)[rotation]
        self._covariance = covariance
        self._pcovariance = self._permute.dot(covariance).dot(self._permute.T)
        cholesky = numpy.linalg.cholesky(self._pcovariance)
        self._fwd_transform = self._permute.T.dot(numpy.linalg.inv(cholesky))
        self._inv_transform = self._permute.T.dot(cholesky)
        self._dist = dist

        super(MeanCovarianceDistribution, self).__init__(
            parameters=dict(mean=mean, covariance=covariance),
            dependencies=dependencies,
            rotation=rotation,
            exclusion=exclusion,
            repr_args=repr_args,
        )

    def get_parameters(self, idx, cache, assert_numerical=True):
        parameters = super(MeanCovarianceDistribution, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)

        mean = parameters["mean"]
        if idx is None:
            return dict(mean=mean, sigma=self._covariance, cache=cache)

        if isinstance(mean, Distribution):
            mean = [mean._get_cache(dim, cache, get=0)
                    for dim in range(len(mean))]
            if any([isinstance(condition, chaospy.Distribution)
                    for condition in mean]):
                raise chaospy.StochasticallyDependentError(
                    "Dangling dependency: %s | %s" % (self, mean))
            mean = numpy.array(mean)
        mean = mean[self._rotation]

        dim = self._rotation.index(idx)
        if dim:
            covinv = numpy.linalg.inv(self._pcovariance[:dim, :dim])
            sigma = numpy.sqrt(
                self._pcovariance[dim, dim]-
                self._pcovariance[dim, :dim].dot(covinv).dot(self._pcovariance[:dim, dim])
            )
            mu_transform = self._pcovariance[dim, :dim].dot(covinv)
            assert numpy.isfinite(sigma), (dim, self._pcovariance)
        else:
            sigma = numpy.sqrt(self._pcovariance[0, 0])
            mu_transform = 0

        return dict(idx=idx, mean=mean, sigma=sigma, dim=dim, mut=mu_transform, cache=cache)


    def _lower(self, idx, mean, sigma, dim, mut, cache):
        return mean[dim]-7.5*numpy.sqrt(self._covariance[idx, idx])

    def _upper(self, idx, mean, sigma, dim, mut, cache):
        return mean[dim]+7.5*numpy.sqrt(self._covariance[idx, idx])

    def _pdf(self, xloc, idx, mean, sigma, dim, mut, cache):
        if dim:
            conditions = [self._get_cache(dim_, cache, get=0)
                          for dim_ in self._rotation[:dim]]
            assert not any([isinstance(condition, chaospy.Distribution)
                            for condition in conditions])
            mean = mean[dim]+mut.dot((numpy.vstack(conditions).T-mean[:dim]).T)
        else:
            mean = mean[dim]
        zloc = (xloc-mean)/sigma
        out = self._dist._get_pdf(zloc, idx, cache)/sigma
        return out

    def _ppf(self, uloc, idx, mean, sigma, dim, mut, cache):
        del sigma
        conditions = [self._get_cache(dim_, cache, get=1)
                      for dim_ in self._rotation[:dim]]
        assert not any([isinstance(condition, chaospy.Distribution)
                        for condition in conditions])
        uloc = numpy.vstack(conditions+[uloc])
        zloc = numpy.zeros(uloc.shape)
        for idx0, idx1 in enumerate(self._rotation[:len(uloc)]):
            zloc[idx0] = self._dist._get_inv(uloc[idx0], idx1, cache)
        xloc = self._inv_transform[idx, :len(uloc)].dot(zloc)+mean[dim]
        return xloc

    def _cdf(self, xloc, idx, mean, sigma, dim, mut, cache):
        del sigma
        conditions = [self[dim_]._get_cache(0, cache, get=0)
                      for dim_ in self._rotation[:dim]]
        assert not any([isinstance(condition, chaospy.Distribution)
                        for condition in conditions])
        xloc = numpy.vstack(conditions+[xloc])
        zloc = self._fwd_transform[idx, :len(xloc)].dot((xloc.T-mean[:len(xloc)]).T)
        uloc = self._dist._get_fwd(zloc, idx, cache)
        return uloc

    def _mom(self, kloc, mean, sigma, cache):
        poly = numpoly.variable(len(self))
        cholesky = numpy.linalg.cholesky(self._covariance)
        poly = numpoly.sum(cholesky*poly, axis=-1)+mean

        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum(self._dist.mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def _ttr(self, kloc, idx, mean, sigma, dim, mut, cache):
        if dim > 1:
            raise chaospy.StochasticallyDependentError(
                "TTR require stochastically independent components.")
        coeff0, coeff1 = self._dist._get_ttr(kloc, idx)
        coeff0 = coeff0*sigma+mean[dim]
        coeff1 = coeff1*sigma*sigma
        return coeff0, coeff1
