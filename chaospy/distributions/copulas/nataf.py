"""Nataf (normal) copula."""
import numpy
from scipy import special
import chaospy

from ..baseclass import CopulaDistribution, Distribution


class nataf(Distribution):
    """Nataf (normal) copula."""

    def __init__(self, covariance, rotation=None):
        covariance = numpy.asarray(covariance)
        assert covariance.ndim == 2, "Covariance must be a matrix"
        assert covariance.shape[0] == covariance.shape[1], (
            "Parameters 'covariance' not a square matrix.")

        dependencies, _, rotation = chaospy.declare_dependencies(
            self,
            parameters=dict(covariance=covariance),
            rotation=rotation,
            dependency_type="accumulate",
        )
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
        )

    def _cdf(self, xloc, idx, cache):
        dim = self._rotation.index(idx)
        conditions = [self._get_cache(dim_, cache, get=0)
                      for dim_ in self._rotation[:dim]]
        assert not any([isinstance(condition, chaospy.Distribution)
                        for condition in conditions])
        xloc = numpy.vstack(conditions+[xloc])
        zloc = self._fwd_transform[idx, :len(xloc)].dot(special.ndtri(xloc))
        out = special.ndtr(zloc)
        return out

    def _ppf(self, qloc, idx, cache):
        dim = self._rotation.index(idx)
        conditions = [self._get_cache(dim_, cache, get=1)
                      for dim_ in self._rotation[:dim]]
        assert not any([isinstance(condition, chaospy.Distribution)
                        for condition in conditions])
        qloc = numpy.vstack(conditions+[qloc])
        zloc = special.ndtri(qloc)
        out = special.ndtr(self._inv_transform[idx, :len(qloc)].dot(zloc))
        return out

    def _pdf(self, xloc, idx, cache):
        raise chaospy.UnsupportedFeature("Copula not supported.")

    def _lower(self, idx, cache):
        return 0.

    def _upper(self, idx, cache):
        return 1.


class Nataf(CopulaDistribution):
    """
    Nataf (normal) copula.

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

    """

    def __init__(self, dist, covariance):
        """
        Args:
            dist (Distribution):
                The distribution to wrap.
            covariance (numpy.ndarray):
                Covariance matrix.
        """
        assert len(dist) == len(covariance)
        return super(Nataf, self).__init__(
            dist=dist,
            trans=nataf(covariance, dist._rotation),
            repr_args=[dist, numpy.array(covariance).tolist()],
        )
