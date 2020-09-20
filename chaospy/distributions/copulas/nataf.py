"""Nataf (normal) copula."""
import numpy
from scipy import special
import chaospy

from ..baseclass import Copula, Index, Distribution


class nataf(Distribution):
    """Nataf (normal) copula."""

    def __init__(self, covariance, rotation):
        covariance = numpy.asarray(covariance)
        assert covariance.ndim == 2, "Covariance must be a matrix"
        assert covariance.shape[0] == covariance.shape[1], (
            "Parameters 'covariance' not a square matrix.")

        accumulant = set()
        dependencies = self._declare_dependencies(len(covariance))
        rotation = numpy.array(rotation)
        for idx in rotation:
            accumulant.add(dependencies[idx])
            dependencies[idx] = accumulant.copy()

        correlation = covariance/numpy.sqrt(numpy.outer(numpy.diag(covariance), numpy.diag(covariance)))
        self._permute = numpy.eye(len(rotation), dtype=int)[rotation]
        self._correlation = self._permute.dot(correlation).dot(self._permute.T)
        cholesky = numpy.linalg.cholesky(self._correlation)
        self._fwd_transform = self._permute.T.dot(numpy.linalg.inv(cholesky))
        self._inv_transform = self._permute.T.dot(cholesky)

        super(nataf, self).__init__(
            parameters=dict(),
            dependencies=dependencies,
            rotation=rotation,
            repr_args=[covariance.tolist()],
            index_cls=NatafIndex,
        )

    def _cdf(self, xloc, cache):
        out = numpy.zeros(xloc.shape)
        for idx in self._rotation:
            xloc_ = xloc[idx].reshape(1, -1)
            out[idx] = self[int(idx)]._get_fwd(xloc_, cache)
        return out

    def _ppf(self, qloc, cache):
        out = numpy.zeros(qloc.shape)
        for idx in self._rotation:
            qloc_ = qloc[idx].reshape(1, -1)
            out[idx] = self[int(idx)]._get_inv(qloc_, cache)
        return out

    def _pdf(self, xloc, cache):
        out = numpy.zeros(xloc.shape)
        for idx in self._rotation:
            xloc_ = xloc[idx].reshape(1, -1)
            out[idx] = self[int(idx)]._get_pdf(xloc_, cache)
        return out

    def _lower(self, cache):
        return numpy.zeros(len(self))

    def _upper(self, cache):
        return numpy.ones(len(self))

    def _cache(self, cache):
        return self


class NatafIndex(Index):

    def __init__(self, parent, conditions=()):
        assert isinstance(parent, nataf)
        super(NatafIndex, self).__init__(
            parent=parent, conditions=conditions)

        idx = parent._rotation[len(conditions)]
        self._correlation = parent._correlation[:len(conditions)+1, :len(conditions)+1]
        self._fwd_transform = parent._fwd_transform[idx]
        self._inv_transform = parent._inv_transform[idx]
        covinv = numpy.linalg.inv(self._correlation[:-1, :-1])
        if conditions:
            self._sigma = numpy.sqrt(
                self._correlation[-1, -1]-
                self._correlation[-1, :-1].dot(covinv).dot(self._correlation[:-1, -1])
            )
        else:
            self._sigma = numpy.sqrt(self._correlation[0, 0])

    def _ppf(self, qloc, idx, parent, conditions, cache):
        assert numpy.any(qloc)
        conditions = [condition._get_cache_2(cache) for condition in conditions]
        qloc = numpy.vstack(conditions+[qloc])
        zloc = special.ndtri(qloc)
        out = special.ndtr(self._inv_transform[:len(qloc)].dot(zloc))
        return out

    def _cdf(self, xloc, idx, parent, conditions, cache):
        conditions = [condition._get_cache_1(cache) for condition in conditions]
        xloc = numpy.vstack(conditions+[xloc])
        zloc = self._fwd_transform[:len(xloc)].dot(special.ndtri(xloc))
        out = special.ndtr(zloc)
        return out

    def _mom(self, kloc, idx, parent, conditions, cache):
        raise chaospy.UnsupportedFeature("Copula not supported.")

    def _ttr(self, kloc, idx, parent, conditions, cache):
        raise chaospy.UnsupportedFeature("Copula not supported.")

    def _cache(self, idx, parent, conditions, cache):
        return self


class Nataf(Copula):
    """
    Nataf (normal) copula.

    Args:
        dist (Distribution):
            The distribution to wrap.
        covariance (numpy.ndarray):
            Covariance matrix.

    Examples:
        >>> distribution = chaospy.Nataf(
        ...     chaospy.Iid(chaospy.Uniform(-1, 1), 2), covariance=[[1, .5], [.5, 1]])
        >>> distribution
        Nataf(Iid(Uniform(lower=-1, upper=1), 2), [[1.0, 0.5], [0.5, 1.0]])
        >>> samples = distribution.sample(3)
        >>> samples.round(4)
        array([[ 0.3072, -0.77  ,  0.9006],
               [ 0.1262,  0.3001,  0.1053]])
        >>> distribution.pdf(samples).round(4)
        array([0.292 , 0.1627, 0.2117])
        >>> distribution.fwd(samples).round(4)
        array([[0.6536, 0.115 , 0.9503],
               [0.4822, 0.8725, 0.2123]])
        >>> mesh = numpy.meshgrid([.4, .5, .6], [.4, .5, .6])
        >>> distribution.inv(mesh).round(4)
        array([[[-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ]],
        <BLANKLINE>
               [[-0.2707, -0.1737, -0.0739],
                [-0.1008,  0.    ,  0.1008],
                [ 0.0739,  0.1737,  0.2707]]])
        >>> distribution.mom([1, 1]).round(4)
        0.1609

    """

    def __init__(self, dist, covariance):
        assert len(dist) == len(covariance)
        return super(Nataf, self).__init__(
            dist=dist,
            trans=nataf(covariance, dist._rotation),
            repr_args=[dist, numpy.array(covariance).tolist()],
        )
