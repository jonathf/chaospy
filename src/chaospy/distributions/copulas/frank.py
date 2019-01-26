"""Frank copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class frank(Archimedean):
    """Frank copula."""

    def __init__(self, length, theta, eps=1e-6):
        assert theta != 0
        self.length = length
        Dist.__init__(self, th=float(theta), eps=eps)

    def __len__(self):
        return self.length

    def gen(self, x, th):
        return -numpy.log((numpy.e**(-th*x)-1)/(numpy.e**-th-1))

    def igen(self, q, th):
        return -numpy.log(1+numpy.e**-q*(numpy.e**-th-1))/th

class Frank(Copula):
    """
    Frank copula.

    Examples:
        >>> distribution = chaospy.Iid(chaospy.Uniform(), 2)
        >>> distribution = chaospy.Frank(distribution)
        >>> print(distribution)
        Frank(Iid(Uniform(lower=0, upper=1), 2), theta=1.0)
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]]
        <BLANKLINE>
         [[0.2101 0.2539 0.3032]
          [0.4391 0.5    0.5609]
          [0.6968 0.7462 0.7899]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[1.1454 1.0054 0.9031]
         [1.0358 1.0207 1.0358]
         [0.9031 1.0055 1.1454]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[0.6536 0.115  0.9503 0.4822]
         [0.8854 0.1587 0.0646 0.395 ]]
        >>> print(numpy.around(distribution.mom((1, 2)), 4))
        0.1804
    """

    def __init__(self, dist, theta=1., eps=1e-4):
        """
        Args:
            dist (Dist) : The Distribution to wrap
            theta (float) : Copula parameter
        """
        self._repr = {"theta": theta}
        return Copula.__init__(self, dist=dist, trans=frank(len(dist), theta, eps))
