"""Ali-Mikhail-Haq copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist


class ali_mikhail_haq(Archimedean):
    """Ali-Mikhail-Haq copula."""

    def __init__(self, length, theta=.5, eps=1e-6):
        assert -1 <= theta < 1
        self.length = length
        Dist.__init__(self, th=float(theta), eps=eps)

    def __len__(self):
        return self.length

    def gen(self, x, th):
        return numpy.log((1-th*(1-x))/x)

    def igen(self, x, th):
        return (1-th)/(numpy.e**x-th)


class AliMikhailHaq(Copula):
    """
    Ali-Mikhail-Haq copula.

    Examples:
        >>> distribution = chaospy.Iid(chaospy.Uniform(), 2)
        >>> distribution = chaospy.AliMikhailHaq(distribution)
        >>> print(distribution)
        AliMikhailHaq(Iid(Uniform(lower=0, upper=1), 2), theta=0.5)
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]]
        <BLANKLINE>
         [[0.2044 0.2634 0.3175]
          [0.4326 0.5099 0.5703]
          [0.6939 0.7533 0.7937]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[1.1636 0.9937 0.9088]
         [1.0285 1.0267 1.0631]
         [0.8882 1.0238 1.1705]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[0.6536 0.115  0.9503 0.4822]
         [0.8886 0.1432 0.0725 0.4046]]
        >>> print(numpy.around(distribution.mom((1, 2)), 4))
        0.1822
    """

    def __init__(self, dist, theta=0.5, eps=1e-6):
        """
        Args:
            dist (Dist) : The Distribution to wrap
            theta (float) : Copula parameter
        """
        self._repr = {"theta": theta}
        trans = ali_mikhail_haq(len(dist), theta, eps)
        return Copula.__init__(self, dist=dist, trans=trans)
