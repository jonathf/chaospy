"""Gumbel Distribution."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist


class gumbel(Archimedean):
    """Gumbel Distribution."""

    def __init__(self, theta=2., eps=1e-6):
        theta = float(theta)
        Dist.__init__(self, th=theta, eps=eps)

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
    """

    def __init__(self, dist, theta, eps=1e-6):
        self._repr = {"theta": theta}
        Copula.__init__(self, dist=dist, trans=gumbel(theta, eps))
