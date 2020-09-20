"""T-Copula."""
import numpy
from scipy import special
import chaospy

from ..baseclass import Copula, Index, Distribution


class t_copula(Distribution):
    """T-Copula."""

    def __init__(self, df, covariance, rotation):
        assert isinstance(df, float)
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
        self._fwd_transform = self._permute.T.dot(numpy.linalg.inv(cholesky)) # CI
        self._inv_transform = self._permute.T.dot(cholesky) # C

        super(t_copula, self).__init__(
            parameters=dict(df=df),
            dependencies=dependencies,
            rotation=rotation,
            repr_args=[covariance.tolist()],
            index_cls=TCopulaIndex,
        )

    def _cdf(self, xloc, df, cache):
        out = numpy.zeros(xloc.shape)
        for idx in self._rotation:
            xloc_ = xloc[idx].reshape(1, -1)
            out[idx] = self[int(idx)]._get_fwd(xloc_, cache)
        return out

    def _ppf(self, qloc, df, cache):
        out = numpy.zeros(qloc.shape)
        for idx in self._rotation:
            qloc_ = qloc[idx].reshape(1, -1)
            out[idx] = self[int(idx)]._get_inv(qloc_, cache)
        return out

    def _pdf(self, xloc, df, cache):
        out = numpy.zeros(xloc.shape)
        for idx in self._rotation:
            xloc_ = xloc[idx].reshape(1, -1)
            out[idx] = self[int(idx)]._get_pdf(xloc_, cache)
        return out

    def _lower(self, df, cache):
        return numpy.zeros(len(self))

    def _upper(self, df, cache):
        return numpy.ones(len(self))

    def _cache(self, df, cache):
        return self


class TCopulaIndex(Index):

    def __init__(self, parent, conditions=()):
        assert isinstance(parent, t_copula)
        super(TCopulaIndex, self).__init__(
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
        df = parent.get_parameters(cache, assert_numerical=True)["df"]
        conditions = [condition._get_cache_2(cache) for condition in conditions]
        qloc = numpy.vstack(conditions+[qloc])
        zloc = special.stdtrit(df, qloc)
        out = special.stdtr(df, self._inv_transform[:len(qloc)].dot(zloc))
        return out

    def _cdf(self, xloc, idx, parent, conditions, cache):
        conditions = [condition._get_cache_1(cache) for condition in conditions]
        df = parent.get_parameters(cache, assert_numerical=True)["df"]
        xloc = numpy.vstack(conditions+[xloc])
        zloc = self._fwd_transform[:len(xloc)].dot(special.stdtrit(df, xloc))
        out = special.stdtr(df, zloc)
        return out

    def _mom(self, kloc, idx, parent, conditons, cache):
        raise chaospy.UnsupportedFeature("Copula not supported.")

    def _ttr(self, kloc, idx, parent, conditons, cache):
        raise chaospy.UnsupportedFeature("Copula not supported.")

    def _cache(self, idx, parent, conditions, cache):
        return self


class TCopula(Copula):
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
        >>> distribution.mom([1, 1]).round(4)
        0.1665

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
