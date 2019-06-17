"""Joe copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class joe(Archimedean):
    """Joe copula."""

    def __init__(self, length, theta, eps=1e-6):
        assert theta >= 1
        self.length = length
        Dist.__init__(self, th=float(theta), eps=eps)

    def __len__(self):
        return self.length

    def gen(self, x, th):
        return -numpy.log(1-(1-x)**th)

    def igen(self, q, th):
        return 1-(1-numpy.e**-q)**(1/th)


class Joe(Copula):
    """
    Joe Copula

    where `theta` is defined on the interval `[1,inf)`.

    Examples:
        >>> distribution = chaospy.Joe(
        ...     chaospy.Iid(chaospy.Uniform(), 2), theta=2)
        >>> print(distribution)
        Joe(Iid(Uniform(lower=0, upper=1), 2), theta=2)
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]]
        <BLANKLINE>
         [[0.1693 0.2351 0.3964]
          [0.3492 0.4459 0.6346]
          [0.5583 0.6497 0.7949]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[1.4454 1.1299 0.8249]
         [1.3193 1.232  1.3345]
         [1.0469 1.1753 1.7359]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[0.6536 0.115  0.9503 0.4822]
         [0.8222 0.1247 0.3292 0.353 ]]
        >>> print(numpy.around(distribution.mom((1, 2)), 4))
        0.2128
    """

    def __init__(self, dist, theta=2., eps=1e-6):
        """
        Args:
            dist (Dist) : The Distribution to wrap
            theta (float) : Copula parameter
        """
        self._repr = {"theta": theta}
        Copula.__init__(self, dist, joe(len(dist), theta, eps))
