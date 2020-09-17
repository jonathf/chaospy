import numpy
from scipy.special import comb
import numpoly
import chaospy

from .conditional import Conditional
from .distribution import Distribution


class MeanCovariance(Distribution):
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
            rotation = [key for key, _ in sorted(enumerate(dist._dependencies), key=lambda x: len(x[1]))]

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

        permute = numpy.zeros((len(rotation), len(rotation)), dtype=int)
        permute[numpy.arange(len(rotation), dtype=int), rotation] = 1

        self._covariance = permute.dot(covariance).dot(permute.T)
        cholesky = numpy.linalg.cholesky(self._covariance)
        self._fwd_transform = permute.T.dot(numpy.linalg.inv(cholesky))
        self._inv_transform = permute.T.dot(cholesky)
        self._conditionals = {}

        self._dist = dist
        self._permute = permute

        super(MeanCovariance, self).__init__(
            parameters=dict(mean=mean, covariance=covariance),
            dependencies=dependencies,
            rotation=rotation,
            exclusion=exclusion,
            repr_args=repr_args,
        )

    def _ppf(self, qloc, mean, covariance, cache):
        out = numpy.zeros(qloc.shape)
        for idx in self._rotation:
            qloc_ = qloc[idx].reshape(1, -1)
            out[idx] = self[idx]._get_inv(qloc_, cache)
        return out

    def _cdf(self, xloc, mean, covariance, cache):
        out = numpy.zeros(xloc.shape)
        for idx in self._rotation:
            xloc_ = xloc[idx].reshape(1, -1)
            out[idx] = self[idx]._get_fwd(xloc_, cache)
        return out

    def _pdf(self, xloc, mean, covariance, cache):
        out = numpy.zeros(xloc.shape)
        for idx in self._rotation:
            xloc_ = xloc[idx].reshape(1, -1)
            out[idx] = self[idx]._get_pdf(xloc_, cache)
        return out

    def _mom(self, kloc, mean, covariance, cache):
        poly = numpoly.variable(len(self))
        cholesky = numpy.linalg.cholesky(covariance)
        poly = numpoly.sum(cholesky*poly, axis=-1)+mean

        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum(self._dist.mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def _ttr(self, kloc, mean, covariance, cache):
        assert numpy.allclose(numpy.diag(numpy.diag(covariance)), covariance), (
            "TTR require stochastically independent components.")
        scale = numpy.sqrt(numpy.diag(covariance))
        coeff0, coeff1 = self._dist._get_ttr(kloc, cache={})
        coeff0 = coeff0*scale+mean
        coeff1 = coeff1*scale*scale
        return coeff0, coeff1

    def _lower(self, mean, covariance, cache):
        if isinstance(mean, Distribution):
            mean = mean._get_lower(cache={})
        if isinstance(covariance, Distribution):
            covariance = covariance._get_lower(cache={})
        std_bound = numpy.sqrt(numpy.diag(covariance))
        return self._dist.lower*std_bound+mean

    def _upper(self, mean, covariance, cache):
        if isinstance(mean, Distribution):
            mean = mean._get_upper(cache={})
        if isinstance(covariance, Distribution):
            covariance = covariance._get_upper(cache={})
        std_bound = numpy.sqrt(numpy.diag(covariance))
        return self._dist.upper*std_bound+mean

    def _cache(self, mean, covariance, cache):
        return self

    def __getitem__(self, index):
        if isinstance(index, int):
            if not -len(self) < index < len(self):
                raise IndexError("index out of bounds: %s" % index)
            if index < 0:
                index += len(self)
            conditions = []
            for idx in self._rotation:
                if idx not in self._conditionals:
                    if conditions:
                        self._conditionals[idx] = MeanCovarianceConditional(
                            parent=self, conditions=chaospy.J(*conditions))
                    else:
                        self._conditionals[idx] = MeanCovarianceConditional(parent=self)
                if idx == index:
                    return self._conditionals[idx]
                conditions.append(self._conditionals[idx])
            return MeanCovarianceConditional(self, chaospy.J(*conditions))
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = len(self) if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            return chaospy.J(*[self[idx] for idx in range(start, stop, step)])
        raise IndexError("unrecognized key")


class MeanCovarianceConditional(Conditional):
    """The conditional structure for the MeanCovariance baseclass."""

    def __init__(self, parent, conditions=()):
        assert isinstance(parent, MeanCovariance)
        idx = parent._rotation[len(conditions)]
        if conditions:
            assert isinstance(conditions, chaospy.J)
            parameters = dict(parent=parent, idx=idx, conditions=conditions)
            repr_args = [parent, conditions]
        else:
            parameters = dict(parent=parent, idx=idx)
            repr_args = [parent]

        super(MeanCovarianceConditional, self).__init__(
            parameters=parameters,
            dependencies=[parent._dependencies[idx].copy()],
            rotation=[0],
            repr_args=repr_args,
        )
        self._covariance = parent._covariance[:len(conditions)+1, :len(conditions)+1]
        self._fwd_transform = parent._fwd_transform[idx]
        self._inv_transform = parent._inv_transform[idx]
        covinv = numpy.linalg.inv(self._covariance[:-1, :-1])
        self._mu_transform = self._covariance[-1, :-1].dot(covinv)
        if conditions:
            self._sigma = numpy.sqrt(
                self._covariance[-1, -1]-
                self._covariance[-1, :-1].dot(covinv).dot(self._covariance[:-1, -1])
            )
        else:
            self._sigma = numpy.sqrt(self._covariance[0, 0])

    def _get_mean(self, parent, cache):
        mean = parent._parameters["mean"]
        if isinstance(mean, Distribution):
            mean = mean._get_cache_1(cache)
            assert not isinstance(mean, Distribution)
        if len(mean) == 1:
            mean = mean*numpy.ones((len(parent),)+mean.shape[1:])
        mean = mean[numpy.array(parent._rotation)]
        return mean

    def _lower(self, idx, parent, conditions, cache):
        out = parent._get_cache_1(cache)
        if isinstance(out, Distribution):
            out = out.lower
        return out[idx]

    def _upper(self, idx, parent, conditions, cache):
        out = parent._get_cache_1(cache)
        if isinstance(out, Distribution):
            out = out.upper
        return out[idx]

    def _ppf(self, uloc, idx, parent, conditions, cache):
        conditions = [condition._get_cache_2(cache) for condition in conditions]
        if any([isinstance(condition, Distribution) for condition in conditions]):
            raise chaospy.UnsupportedFeature("Conditions not resolved!")
        uloc = numpy.vstack(conditions+[uloc])
        mean = self._get_mean(parent, cache)
        zloc = parent._dist[parent._rotation[:len(uloc)]]._get_inv(uloc, cache)
        xloc = (self._inv_transform[:len(uloc)].dot(zloc)+mean[len(uloc)-1])
        return xloc

    def _cdf(self, xloc, idx, parent, conditions, cache):
        conditions = [condition._get_cache_1(cache) for condition in conditions]
        if any([isinstance(condition, Distribution) for condition in conditions]):
            raise chaospy.UnsupportedFeature("Conditions not resolved!")
        mean = self._get_mean(parent, cache)
        xloc = numpy.vstack(conditions+[xloc])
        zloc = self._fwd_transform[:len(xloc)].dot((xloc[:len(xloc)].T-mean[:len(xloc)]).T)
        uloc = parent._dist[int(idx)]._get_fwd(zloc[numpy.newaxis], cache)
        return uloc

    def _pdf(self, xloc, idx, parent, conditions, cache):
        conditions = [condition._get_cache_1(cache) for condition in conditions]
        if any([isinstance(condition, Distribution) for condition in conditions]):
            raise chaospy.UnsupportedFeature("Conditions not resolved!")
        xloc = numpy.vstack(conditions+[xloc])
        mean = self._get_mean(parent, cache)
        mu = mean[len(conditions)]
        if conditions:
            mu += self._mu_transform.dot((xloc[:-1].T-mean[:len(conditions)].T).T)
        zloc = (xloc[-1]-mu)/self._sigma
        out = parent._dist[int(idx)]._get_pdf(zloc[numpy.newaxis], cache)/self._sigma
        return out

    def _mom(self, kloc, idx, parent, conditions, cache):
        if conditions:
            raise chaospy.UnsupportedFeature(
                "Analytical moment of a conditional not supported")
        mean = self._get_mean(parent, cache)
        poly = numpoly.variable(1)
        poly = numpoly.sum(self._sigma*poly, axis=-1)+mean[len(conditions)]
        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum(parent._dist[int(idx)]._get_mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def _ttr(self, kloc, idx, parent, conditions, cache):
        mean = self._get_mean(parent, cache)
        coeff0, coeff1 = parent._dist[int(idx)]._get_ttr(kloc)
        coeff0 = coeff0*self._sigma+mean[len(conditions)]
        coeff1 = coeff1*self._sigma*self._sigma
        return coeff0, coeff1

    def _cache(self, idx, parent, conditions, cache):
        return self
