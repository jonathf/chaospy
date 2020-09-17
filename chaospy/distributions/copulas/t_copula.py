"""T-Copula."""
import numpy
from scipy import special

from ..baseclass import Copula, Distribution


class t_copula(Distribution):
    """T-Copula."""

    def __init__(self, df, R, rotation=None):
        if rotation is None:
            rotation = range(len(R))
        rotation = numpy.array(rotation)

        accumulant = set()
        dependencies = self._declare_dependencies(len(R))
        for idx in rotation:
            accumulant.add(dependencies[idx])
            dependencies[idx] = accumulant.copy()

        P = numpy.eye(len(R))[rotation]
        R = numpy.dot(P, numpy.dot(R, P.T))
        R = numpy.linalg.cholesky(R)
        R = numpy.dot(P.T, numpy.dot(R, P))
        Ci = numpy.linalg.inv(R)
        super(t_copula, self).__init__(
            parameters=dict(df=df, C=R, Ci=Ci),
            rotation=rotation,
            dependencies=dependencies,
        )

    def _cdf(self, x, df, C, Ci, cache):
        out = special.stdtr(df, numpy.dot(Ci, special.stdtrit(df, x)))
        return out

    def _ppf(self, q, df, C, Ci, cache):
        out = special.stdtr(df, numpy.dot(C, special.stdtrit(df, q)))
        return out

    def _lower(self, df, C, Ci, cache):
        return numpy.zeros(len(self))

    def _upper(self, df, C, Ci, cache):
        return numpy.ones(len(self))

    def _cache(self, df, C, Ci, cache):
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
        ...     df=5, R=[[1, .5], [.5, 1]])
        >>> distribution
        TCopula(Iid(Uniform(lower=-1, upper=1), 2), df=5, R=[[1, 0.5], [0.5, 1]])
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

    def __init__(self, dist, df, R, rotation=None):
        self._repr = {"df": df, "R": R}
        super(TCopula, self).__init__(
            dist=dist,
            trans=t_copula(df, R),
            rotation=rotation,
            repr_args=[dist, "df=%s" % df, "R=%s" % R],
        )
