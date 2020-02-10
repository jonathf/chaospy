"""T-Copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
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

    Examples:
        >>> distribution = chaospy.Iid(chaospy.Uniform(), 2)
        >>> R = [[1, 0.5], [0.5, 1]]
        >>> copula = chaospy.TCopula(distribution, 5, R)
        >>> print(copula)
        TCopula(Iid(Uniform(lower=0, upper=1), 2), R=[[1, 0.5], [0.5, 1]], df=5)
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(copula.inv(mesh), 4))
        [[[0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]]
        <BLANKLINE>
         [[0.1832 0.2784 0.4004]
          [0.3656 0.5    0.6344]
          [0.5996 0.7216 0.8168]]]
        >>> print(numpy.around(copula.fwd(copula.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(copula.pdf(copula.inv(mesh)), 4))
        [[1.4656 1.0739 0.8912]
         [1.2486 1.1547 1.2486]
         [0.8912 1.0739 1.4656]]
        >>> print(numpy.around(copula.sample(4), 4))
        [[0.6536 0.115  0.9503 0.4822]
         [0.8783 0.1053 0.2107 0.4021]]
    """

    def __init__(self, dist, df, R):
        """
        Args:
            dist (Dist):
                The Distribution to wrap in a copula.
            R (numpy.ndarray):
                Covariance matrix defining dependencies..
            df (float):
                The degree of freedom in the underlying student-t distribution.
        """
        self._repr = {"df": df, "R": R}
        Copula.__init__(self, dist=dist, trans=t_copula(df, R))
