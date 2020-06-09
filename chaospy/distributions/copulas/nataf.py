"""Nataf (normal) copula."""
import numpy
from scipy import special

from .baseclass import Copula
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

    Args:
        dist (Dist):
            The Distribution to wrap.
        R (numpy.ndarray):
            Covariance matrix.

    Examples:
        >>> distribution = chaospy.Nataf(
        ...     chaospy.Iid(chaospy.Uniform(-1, 1), 2), R=[[1, .5], [.5, 1]])
        >>> distribution
        Nataf(Iid(Uniform(lower=-1, upper=1), 2), R=[[1, 0.5], [0.5, 1]])
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

    def __init__(self, dist, R, ordering=None):
        self._repr = {"R": R}
        assert len(dist) == len(R)
        return Copula.__init__(self, dist=dist, trans=nataf(R, ordering))
