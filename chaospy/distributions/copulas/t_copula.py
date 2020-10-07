"""T-Copula."""
import numpy
from scipy import special
import chaospy

from ..baseclass import CopulaDistribution, Distribution


class t_copula(Distribution):
    """T-Copula."""

    def __init__(self, df, covariance, rotation=None):
        assert isinstance(df, float)
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
        correlation = covariance/numpy.sqrt(numpy.outer(
            numpy.diag(covariance), numpy.diag(covariance)))
        self._permute = numpy.eye(len(rotation), dtype=int)[rotation]
        self._correlation = self._permute.dot(correlation).dot(self._permute.T)
        cholesky = numpy.linalg.cholesky(self._correlation)
        self._fwd_transform = self._permute.T.dot(numpy.linalg.inv(cholesky))
        self._inv_transform = self._permute.T.dot(cholesky)

        super(t_copula, self).__init__(
            parameters=dict(df=df),
            dependencies=dependencies,
            rotation=rotation,
            repr_args=[covariance.tolist()],
        )

    def _cdf(self, xloc, idx, df, cache):
        dim = self._rotation.index(idx)
        conditions = [self._get_cache(dim_, cache, get=0)
                      for dim_ in self._rotation[:dim]]
        assert not any([isinstance(condition, chaospy.Distribution)
                        for condition in conditions])
        xloc = numpy.vstack(conditions+[xloc])
        zloc = self._fwd_transform[idx, :len(xloc)].dot(special.stdtrit(df, xloc))
        out = special.stdtr(df, zloc)
        return out

    def _ppf(self, qloc, idx, df, cache):
        dim = self._rotation.index(idx)
        conditions = [self._get_cache(dim_, cache, get=1)
                      for dim_ in self._rotation[:dim]]
        assert not any([isinstance(condition, chaospy.Distribution)
                        for condition in conditions])
        qloc = numpy.vstack(conditions+[qloc])
        zloc = special.stdtrit(df, qloc)
        out = special.stdtr(df, self._inv_transform[idx, :len(qloc)].dot(zloc))
        return out

    def _pdf(self, xloc, idx, df, cache):
        raise chaospy.UnsupportedFeature("Copula not supported.")

    def _lower(self, idx, df, cache):
        return 0.

    def _upper(self, idx, df, cache):
        return 1.


class TCopula(CopulaDistribution):
    """
    T-Copula.

    Args:
        dist (Distribution):
            The distribution to wrap in a copula.
        R (numpy.ndarray):
            Covariance matrix defining dependencies..
        df (float):
            The degree of freedom in the underlying student-t distribution.

    Examples:
        >>> distribution = chaospy.TCopula(
        ...     chaospy.Iid(chaospy.Uniform(-1, 1), 2),
        ...     df=5, covariance=[[1, .5], [.5, 1]])
        >>> distribution
        TCopula(Iid(Uniform(lower=-1, upper=1), 2), 5.0, [[1.0, 0.5], [0.5, 1.0]])
        >>> samples = distribution.sample(3)
        >>> samples.round(4)
        array([[ 0.3072, -0.77  ,  0.9006],
               [ 0.1274,  0.3147,  0.1928]])
        >>> distribution.pdf(samples).round(4)
        array([0.2932, 0.1367, 0.1969])
        >>> distribution.fwd(samples).round(4)
        array([[0.6536, 0.115 , 0.9503],
               [0.4822, 0.8725, 0.2123]])
        >>> mesh = numpy.meshgrid([.4, .5, .6], [.4, .5, .6])
        >>> distribution.inv(mesh).round(4)
        array([[[-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ]],
        <BLANKLINE>
               [[-0.2699, -0.1738, -0.0741],
                [-0.1011,  0.    ,  0.1011],
                [ 0.0741,  0.1738,  0.2699]]])

    """

    def __init__(self, dist, df, covariance):
        assert len(dist) == len(covariance)
        df = float(df)
        covariance = numpy.asfarray(covariance)
        super(TCopula, self).__init__(
            dist=dist,
            trans=t_copula(df, covariance, dist._rotation),
            repr_args=[dist, df, covariance.tolist()],
        )
