"""Clayton copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class clayton(Archimedean):
    """Clayton copula."""

    def __init__(self, length, theta=1., eps=1e-6):
        self.length = length
        Dist.__init__(self, th=float(theta), eps=eps)

    def __len__(self):
        return self.length

    def gen(self, x, th):
        return (x**-th-1)/th

    def igen(self, x, th):
        return (1.+th*x)**(-1./th)


class Clayton(Copula):
    """
    Clayton Copula.

    Args:
        dist (Dist) : The Distribution to wrap
        theta (float) : Copula parameter

    Returns:
        (Dist) : The resulting copula distribution.

    Examples:
        >>> distribution = chaospy.Clayton(
        ...     chaospy.Iid(chaospy.Uniform(), 2), theta=2)
        >>> print(distribution)
        Clayton(Iid(Uniform(lower=0, upper=1), 2), theta=2)
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]]
        <BLANKLINE>
         [[0.1987 0.3758 0.5197]
          [0.3101 0.5464 0.6994]
          [0.4777 0.7361 0.8525]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[2.3697 1.4016 1.1925]
         [1.9803 1.4482 1.5538]
         [1.0651 1.1642 1.6861]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[0.6536 0.115  0.9503 0.4822]
         [0.9043 0.0852 0.3288 0.4633]]
        >>> print(numpy.around(distribution.mom((1, 2)), 4))
        0.2196
    """

    def __init__(self, dist, theta=2., eps=1e-6):
        """
        Args:
            dist (Dist) : The Distribution to wrap
            theta (float) : Copula parameter
        """
        self._repr = {"theta": theta}
        trans = clayton(len(dist), theta=theta, eps=eps)
        return Copula.__init__(self, dist=dist, trans=trans)
