"""Gumbel Distribution."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist


class gumbel(Archimedean):
    """Gumbel Distribution."""

    def __init__(self, length, theta=2., eps=1e-6):
        self.length = length
        Dist.__init__(self, th=float(theta), eps=eps)

    def __len__(self):
        return self.length

    def gen(self, x, th):
        return (-numpy.log(x))**th

    def igen(self, x, th):
        return numpy.e**(-x**th)


class Gumbel(Copula):
    r"""
    Gumbel Copula

    .. math::
        \phi(x;th) = \frac{x^{-th}-1}{th}
        \phi^{-1}(q;th) = (q*th + 1)^{-1/th}

    where `th` (or theta) is defined on the interval `[1,inf)`.

    Args:
        dist (Dist) : The Distribution to wrap
        theta (float) : Copula parameter

    Returns:
        (Dist) : The resulting copula distribution.

    Examples:
        >>> distribution = chaospy.Gumbel(
        ...     chaospy.Iid(chaospy.Uniform(), 2), theta=2)
        >>> print(distribution)
        Gumbel(Iid(Uniform(lower=0, upper=1), 2), theta=2)
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]
          [0.25   0.5    0.75  ]]
        <BLANKLINE>
         [[0.5439 0.3319 0.2369]
          [0.6426 0.3802 0.2534]
          [0.7482 0.4292 0.2653]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[ 2.3224  4.6584 11.7092]
         [ 2.5896  5.4078 18.5493]
         [ 2.0433  4.601  23.5876]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[0.6536 0.115  0.9503 0.4822]
         [0.3151 0.665  0.1675 0.3745]]
    """

    def __init__(self, dist, theta, eps=1e-6):
        """
        Args:
            dist (Dist) : The Distribution to wrap
            theta (float) : Copula parameter
        """
        self._repr = {"theta": theta}
        Copula.__init__(self, dist=dist, trans=gumbel(len(dist), theta, eps))
