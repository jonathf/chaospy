"""Nataf (normal) copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class nataf(Dist):
    """Nataf (normal) copula."""

    def __init__(self, R, ordering=None):
        if ordering is None:
            ordering = range(len(R))
        ordering = numpy.array(ordering)
        P = numpy.eye(len(R))[ordering]
        R = numpy.dot(P, numpy.dot(R, P.T))
        R = numpy.linalg.cholesky(R)
        R = numpy.dot(P.T, numpy.dot(R, P))
        Ci = numpy.linalg.inv(R)
        Dist.__init__(self, C=R, Ci=Ci)
        self.length = len(R)

    def __len__(self):
        return self.length

    def _cdf(self, x, C, Ci):
        out = special.ndtr(numpy.dot(Ci, special.ndtri(x)))
        return out

    def _ppf(self, q, C, Ci):
        out = special.ndtr(numpy.dot(C, special.ndtri(q)))
        return out

    def _lower(self, C, Ci):
        return 0.

    def _upper(self, C, Ci):
        return 1.


class Nataf(Copula):
    """
    Nataf (normal) copula.

    Examples:
        >>> distribution = chaospy.Iid(chaospy.Uniform(), 2)
        >>> R = [[1, 0.5], [0.5, 1]]
        >>> copula = chaospy.Nataf(distribution, R)
        >>> print(copula)
        Nataf(Iid(Uniform(lower=0, upper=1), 2), R=[[1, 0.5], [0.5, 1]])
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(copula.inv(mesh), 4))
        [[[0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]]
        <BLANKLINE>
         [[0.1784 0.2796 0.4025]
          [0.368  0.5    0.632 ]
          [0.5975 0.7204 0.8216]]]
        >>> print(numpy.around(copula.fwd(copula.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(copula.pdf(copula.inv(mesh)), 4))
        [[1.4061 1.0909 0.9482]
         [1.2223 1.1547 1.2223]
         [0.9482 1.0909 1.4061]]
        >>> print(numpy.around(copula.sample(4), 4))
        [[0.6536 0.115  0.9503 0.4822]
         [0.8816 0.0983 0.2466 0.4021]]
    """

    def __init__(self, dist, R, ordering=None):
        """
        Args:
            dist (Dist) : The Distribution to wrap.
            R (numpy.ndarray) : Covariance matrix.
        """
        self._repr = {"R": R}
        assert len(dist) == len(R)
        return Copula.__init__(self, dist=dist, trans=nataf(R, ordering))
