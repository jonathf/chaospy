"""Multivariate Log-Normal Distribution."""
import numpy
from scipy import special
import chaospy

from ..baseclass import Distribution


class MvLogNormal(Distribution):
    """
    Multivariate Log-Normal Distribution.

    Args:
        loc (float, Distribution):
            Mean vector
        scale (float, Distribution):
            Covariance matrix or variance vector if scale is a 1-d vector.

    Examples:
        >>> distribution = chaospy.MvLogNormal([1, 2], [[1, 0.6], [0.6, 1]])
        >>> distribution
        MvLogNormal([1, 2], [[1, 0.6], [0.6, 1]])
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> distribution.inv(mesh).round(4)
        array([[[ 1.3847,  2.7183,  5.3361],
                [ 1.3847,  2.7183,  5.3361],
                [ 1.3847,  2.7183,  5.3361]],
        <BLANKLINE>
               [[ 2.874 ,  4.3077,  6.4566],
                [ 4.9298,  7.3891, 11.075 ],
                [ 8.4562, 12.6745, 18.9971]]])
        >>> distribution.fwd(distribution.inv(mesh)).round(4)
        array([[[0.25, 0.5 , 0.75],
                [0.25, 0.5 , 0.75],
                [0.25, 0.5 , 0.75]],
        <BLANKLINE>
               [[0.25, 0.25, 0.25],
                [0.5 , 0.5 , 0.5 ],
                [0.75, 0.75, 0.75]]])
        >>> distribution.pdf(distribution.inv(mesh)).round(4)
        array([[0.0108, 0.002 , 0.    ],
               [0.0135, 0.0035, 0.    ],
               [0.0107, 0.0038, 0.0001]])
        >>> distribution.sample(4).round(4)
        array([[ 4.0351,  0.8185, 14.1201,  2.5996],
               [23.279 ,  1.8986,  4.9261,  5.8399]])
        >>> distribution.mom((1, 2)).round(4)
        6002.9122
    """

    def __init__(
            self,
            mu,
            sigma,
            rotation=None,
    ):
        repr_args = [mu, sigma]
        dependencies, parameters, rotation = chaospy.declare_dependencies(
            self,
            parameters=dict(mu=mu, sigma=sigma),
            rotation=rotation,
            dependency_type="accumulate",
        )

        sigma = parameters["sigma"]
        assert sigma.ndim == 2, (
            "Covariance must either be scalar, vector or matrix")

        self._permute = numpy.eye(len(rotation), dtype=int)[rotation]
        self._psigma = self._permute.dot(sigma).dot(self._permute.T)
        cholesky = numpy.linalg.cholesky(self._psigma)
        self._fwd_transform = self._permute.T.dot(numpy.linalg.inv(cholesky))
        self._inv_transform = self._permute.T.dot(cholesky)

        super(MvLogNormal, self).__init__(
            parameters=parameters,
            dependencies=dependencies,
            rotation=rotation,
            repr_args=repr_args,
        )

    def get_parameters(self, idx, cache, assert_numerical=True):
        parameters = super(MvLogNormal, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        if idx is None:
            parameters.pop("idx")
        return parameters

    def _pdf(self, xloc, idx, mu, sigma, cache):
        dim = self._rotation.index(idx)
        if dim:
            covinv = numpy.linalg.inv(self._psigma[:dim, :dim])
            mu_transform = self._psigma[dim, :dim].dot(covinv)
            assert numpy.isfinite(sigma).all(), (dim, self._psigma)
            conditions = [self._get_cache(dim_, cache, get=0)
                          for dim_ in self._rotation[:dim]]
            assert not any([isinstance(condition, chaospy.Distribution)
                            for condition in conditions])
            mu = mu[numpy.asarray(self._rotation[:dim+1])]
            mu = mu[dim]+mu_transform.dot((numpy.vstack(conditions).T-mu[:dim]).T)
            sigma = numpy.sqrt(
                self._psigma[dim, dim]-
                self._psigma[dim, :dim].dot(covinv).dot(self._psigma[:dim, dim])
            )
        else:
            mu = mu[dim]
            sigma = numpy.sqrt(self._psigma[0, 0])

        out = numpy.zeros(xloc.shape)
        indices = xloc > 0
        zloc = (numpy.log(xloc[indices])-mu)/sigma
        out[indices] = 1/numpy.sqrt(2*numpy.pi)*numpy.exp(-zloc**2/2.)/xloc[indices]
        return out

    def _cdf(self, xloc, idx, mu, sigma, cache):
        dim = self._rotation.index(idx)
        conditions = [self._get_cache(dim_, cache, get=0)
                      for dim_ in self._rotation[:dim]]
        assert not any([isinstance(condition, chaospy.Distribution)
                        for condition in conditions])
        yloc = numpy.vstack(conditions+[xloc])
        yloc = numpy.log(numpy.abs(yloc) + numpy.where(yloc <= 0, 1., 0.))
        mu = mu[numpy.asarray(self._rotation[:len(yloc)])]
        loc = (yloc.T-mu).T
        zloc = self._fwd_transform[idx, :len(yloc)].dot(loc)
        out = special.ndtr(zloc)
        return numpy.where(xloc <= 0, 0., out)

    def _ppf(self, uloc, idx, mu, sigma, cache):
        dim = self._rotation.index(idx)
        conditions = [self._get_cache(dim_, cache, get=1)
                      for dim_ in self._rotation[:dim]]
        assert not any([isinstance(condition, chaospy.Distribution)
                        for condition in conditions])
        uloc = numpy.vstack(conditions+[uloc])
        zloc = special.ndtri(uloc)
        loc = self._inv_transform[idx, :len(uloc)].dot(zloc)
        xloc = loc+mu[idx]
        out = numpy.e**xloc
        return out

    def _mom(self, kloc, mu, sigma, cache):
        output =  numpy.dot(kloc, mu)
        output += .5*numpy.dot(numpy.dot(kloc, sigma), kloc)
        output =  numpy.e**(output)
        return output

    def _lower(self, idx, mu, sigma, cache):
        return 0.

    def _upper(self, idx, mu, sigma, cache):
        return numpy.exp(7.1*numpy.sqrt(sigma[idx, idx])) + mu[idx]
