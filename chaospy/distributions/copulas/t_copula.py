"""T-Copula."""
import numpy
from scipy import special

from .baseclass import Copula
from ..baseclass import Dist


class t_copula(Dist):
    """T-Copula."""

    def __init__(self, df, R):
        C = numpy.linalg.cholesky(R)
        Ci = numpy.linalg.inv(C)
        self.length = len(R)
        Dist.__init__(self, df=df, C=C, Ci=Ci)

    def __len__(self):
        return self.length

    def _cdf(self, x, df, C, Ci):
        out = special.stdtr(df, numpy.dot(Ci, special.stdtrit(df, x)))
        return out

    def _ppf(self, q, df, C, Ci):
        out = special.stdtr(df, numpy.dot(C, special.stdtrit(df, q)))
        return out

    def _lower(self, df, C, Ci):
        return 0.

    def _upper(self, df, C, Ci):
        return 1.


class TCopula(Copula):
    """
    T-Copula.

    Args:
        dist (Dist):
            The Distribution to wrap in a copula.
        R (numpy.ndarray):
            Covariance matrix defining dependencies..
        df (float):
            The degree of freedom in the underlying student-t distribution.

    Examples:
        >>> distribution = chaospy.TCopula(
        ...     chaospy.Iid(chaospy.Uniform(-1, 1), 2),
        ...     df=5, R=[[1, .5], [.5, 1]])
        >>> distribution
        TCopula(Iid(Uniform(lower=-1, upper=1), 2), R=[[1, 0.5], [0.5, 1]], df=5)
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

    def __init__(self, dist, df, R):
        self._repr = {"df": df, "R": R}
        Copula.__init__(self, dist=dist, trans=t_copula(df, R))
