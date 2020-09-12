import numpy
from scipy.special import comb
import numpoly
import chaospy

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

    def __init__(self, dist, mean=0, covariance=1, rotation=None, repr_args=None):
        assert isinstance(dist, Distribution), "'dist' should be a distribution"
        mean = mean if isinstance(mean, Distribution) else numpy.atleast_1d(mean)
        self._cov_is_dist = isinstance(covariance, Distribution)
        covariance = covariance if self._cov_is_dist else numpy.atleast_1d(covariance)
        length = max(len(dist), len(mean), len(covariance))

        if isinstance(mean, Distribution):
            if len(mean) == 1 and length > 1:
                mean = chaospy.Iid(mean, length)
        else:
            assert mean.ndim == 1, "Parameter 'mean' have too many dimensions"
            if len(mean) == 1 and length > 1:
                mean = numpy.repeat(mean.item(), length)
        assert len(mean) == length

        if len(dist) == 1 and length > 1:
            dist = chaospy.Iid(dist, length)

        if self._cov_is_dist:
            if len(covariance) == 1 and length > 1:
                covariance = chaospy.Iid(covariance, length)
        else:
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

        self._dist = dist
        rotation, dependencies, permute = self._rotate(rotation)
        self._permute = permute
        super(MeanCovariance, self).__init__(
            parameters=dict(mean=mean, covariance=covariance),
            dependencies=dependencies,
            rotation=rotation,
            repr_args=repr_args,
        )

    def _rotate(self, rotation=None):
        if rotation is not None:
            rotation = list(rotation)
        else:
            rotation = [key for key, _ in sorted(enumerate(self._dist._dependencies), key=lambda x: len(x[1]))]

        accumulant = set()
        dependencies = [deps.copy() for deps in self._dist._dependencies]
        for idx in rotation:
            accumulant.update(self._dist._dependencies[idx])
            dependencies[idx] = accumulant.copy()

        permute = numpy.zeros((len(rotation), len(rotation)), dtype=int)
        permute[numpy.arange(len(rotation), dtype=int), rotation] = 1
        return rotation, dependencies, permute

    def _ppf(self, qloc, mean, covariance, cache):
        if not self._cov_is_dist:
            covariance = self._permute.dot(covariance).dot(self._permute.T)
            cholesky = numpy.linalg.cholesky(covariance)
            transform = self._permute.T.dot(cholesky).dot(self._permute)
        zloc = self._dist.inv(qloc)
        out = (transform.dot(zloc).T+mean).T
        return out

    def _cdf(self, xloc, mean, covariance, cache):
        if not self._cov_is_dist:
            covariance = self._permute.dot(covariance).dot(self._permute.T)
            cholesky = numpy.linalg.cholesky(covariance)
            transform = self._permute.T.dot(numpy.linalg.inv(cholesky)).dot(self._permute)
        zloc = transform.dot((xloc.T-mean).T)
        out = self._dist.fwd(zloc)
        return out

    def _pdf(self, xloc, mean, covariance, cache):
        if not self._cov_is_dist:
            covariance = self._permute.dot(covariance).dot(self._permute.T)
        xloc = self._permute.dot(xloc)
        mean = self._permute.dot(mean)

        out = numpy.ones(xloc.shape)
        sigma = numpy.sqrt(covariance[0, 0])
        zloc = (xloc.T-mean).T/sigma
        out[0] = (self._dist.pdf(zloc, decompose=True)/sigma)[0]

        for idx in range(1, len(self)):
            covinv = numpy.linalg.inv(covariance[:idx, :idx])
            mu_ = mean[idx]+covariance[idx, :idx].dot(covinv).dot((xloc[:idx].T-mean[:idx]).T)
            sigma = numpy.sqrt(covariance[idx, idx]-covariance[idx, :idx].dot(covinv).dot(covariance[:idx, idx]))
            zloc[idx] = (xloc[idx]-mu_)/sigma
            out[idx] = (self._dist.pdf(zloc, decompose=True)/sigma)[idx]

        out = self._permute.T.dot(out)
        return out

    def _mom(self, kloc, mean, covariance, cache):

        poly = numpoly.variable(len(self))
        if not self._cov_is_dist:
            cholesky = numpy.linalg.cholesky(covariance)
            poly = numpoly.sum(cholesky*poly, axis=-1)+mean

        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum(self._dist.mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def _ttr(self, kloc, mean, covariance, cache):
        assert not self._cov_is_dist
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

    def _value(self, mean, covariance, cache):
        return self
