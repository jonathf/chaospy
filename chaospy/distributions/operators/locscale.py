import numpy
from scipy.special import comb
import numpoly
import chaospy

from ..baseclass import Dist
from .. import evaluation


class LocScaling(Dist):
    r"""
    Location-Scaling transformation.

    Transforms any i.i.d. distributions with zero mean and unit variance
    into provided `mean` and `covariance`. The transformation works on other
    distributions as well, but in those cases `mean` and `covariance` are
    likely not correct.

    Args:
        dist (Dist):
            The underlying distribution to be scaled.
        mean (float, Sequence[float], Dist):
            Mean vector.
        covariance (float, Sequence[Sequence[float]], Dist):
            Covariance matrix or variance vector if scale is a 1-d vector.
            If omitted, assumed to be 1.
        rotation (Sequence[int], Sequence[Sequence[bool]]):
            The order of which to resolve conditionals. Either as a sequence of
            column rotations, or as a permutation matrix.
            Defaults to `range(len(distribution))` which is the same as
            `p(x0), p(x1|x0), p(x2|x0,x1), ...`.

    Attributes:
        mean (numpy.ndarray):
            The mean of the distribution.
        covariance (numpy.ndarray):
            The covariance of the distribution.

    Examples:
        >>> n = chaospy.distributions.collection.normal.normal
        >>> dist = LocScaling(n(), [-10, 0, 10], [[1, .2, .3], [.2, 1, .4], [.3, .4, 1]])
        >>> chaospy.Cov(dist)
        array([[1. , 0.2, 0.3],
               [0.2, 1. , 0.4],
               [0.3, 0.4, 1. ]])
        >>> chaospy.E(dist)
        array([-10.,   0.,  10.])

    """

    @property
    def mean(self):
        return self._mean.copy()

    @property
    def covariance(self):
        return self._covariance.copy()

    def __init__(self, dist, mean, covariance=None, rotation=None):
        mean = numpy.atleast_1d(mean)
        assert mean.ndim == 1, "Parameter 'mean' have too many dimensions"

        if len(dist) == 1 and len(mean) > 1:
            dist = chaospy.Iid(dist, len(mean))
        elif len(dist) > 1 and len(mean) == 1:
            mean = numpy.repeat(mean, len(dist))

        if covariance is None:
            covariance = numpy.diag(numpy.ones(len(mean)))
        else:
            covariance = numpy.asarray(covariance)

        assert covariance.ndim <= 2, (
            "Covariance must either be scalar, vector or matrix")
        if covariance.ndim == 0:
            covariance = numpy.eye(len(mean))*covariance
        elif covariance.ndim == 1:
            covariance = numpy.diag(covariance)
        assert covariance.shape == (len(mean), len(mean)), (
            "Parameters 'mean' and 'covariance' have shape mismatch.")

        self._dist = dist
        self._mean = mean
        self._covariance = covariance
        assert len(self._mean) == len(self._covariance)

        if rotation is not None:
            self._rotation = numpy.asarray(rotation, dtype=int)
        else:
            self._rotation = [key for key, _ in sorted(enumerate(dist._dependencies), key=lambda x: len(x[1]))]

        accumulant = set()
        self._dependencies = [None]*len(dist)
        for idx in self._rotation:
            accumulant.update(dist._dependencies[idx])
            self._dependencies[idx] = accumulant.copy()

        self._std_bound = numpy.sqrt(numpy.diag(self._covariance))
        self._permute = numpy.zeros((len(self._covariance), len(self._covariance)), dtype=int)
        self._permute[numpy.arange(len(self._covariance), dtype=int), self._rotation] = 1

        covariance = self._permute.T.dot(self._covariance).dot(self._permute)
        chol = numpy.linalg.cholesky(covariance)
        self._ppf_transform = self._permute.dot(chol).dot(self._permute.T)
        self._cdf_transform = self._permute.dot(numpy.linalg.inv(chol)).dot(self._permute.T)

        poly = numpoly.variable(len(self))
        chol = numpy.linalg.cholesky(self._covariance)
        self._poly = poly = numpoly.sum(chol*poly, axis=-1)+self._mean

        Dist.__init__(self)

    def _ppf(self, qloc):
        zloc = self._dist.inv(qloc)
        out = (self._ppf_transform.dot(zloc).T+self._mean).T
        return out

    def _cdf(self, xloc):
        zloc = self._cdf_transform.dot((xloc.T-self._mean).T)
        out = self._dist.fwd(zloc)
        return out

    def _pdf(self, xloc):
        xloc = self._permute.dot(xloc)
        mean = self._permute.dot(self._mean)
        cov = self._permute.dot(self._covariance).dot(self._permute.T)

        out = numpy.ones(xloc.shape)
        sigma = numpy.sqrt(cov[0, 0])
        zloc = (xloc.T-mean).T/sigma
        out[0] = (self._dist.pdf(zloc, decompose=True)/sigma)[0]

        for idx in range(1, len(self)):
            covinv = numpy.linalg.inv(cov[:idx, :idx])
            mu_ = mean[idx]+cov[idx, :idx].dot(covinv).dot((xloc[:idx].T-mean[:idx]).T)
            sigma = numpy.sqrt(cov[idx, idx]-cov[idx, :idx].dot(covinv).dot(cov[:idx, idx]))
            zloc[idx] = (xloc[idx]-mu_)/sigma
            out[idx] = (self._dist.pdf(zloc, decompose=True)/sigma)[idx]

        out = self._permute.T.dot(out)
        return out

    def _mom(self, kloc):
        poly = numpoly.set_dimensions(numpoly.prod(self._poly**kloc), len(self))
        out = sum(self._dist.mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def _ttr(self, kloc):
        assert numpy.allclose(numpy.diag(numpy.diag(self._covariance)), self._covariance), (
            "TTR require stochastically independent components.")
        coeff0, coeff1 = evaluation.evaluate_recurrence_coefficients(self._dist, kloc)
        scale = numpy.sqrt(numpy.diag(self._covariance))
        coeff0 = coeff0*scale+self._location
        coeff1 = coeff1*scale*scale
        return coeff0, coeff1

    def _lower(self):
        return self._dist.lower*self._std_bound+self._mean

    def _upper(self):
        return self._dist.upper*self._std_bound+self._mean
