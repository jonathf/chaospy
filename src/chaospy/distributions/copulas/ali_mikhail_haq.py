"""Ali-Mikhail-Haq copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class ali_mikhail_haq(Archimedean):
    """Ali-Mikhail-Haq copula."""

    def __init__(self, N, theta=.5, eps=1e-6):
        theta = float(theta)
        assert -1 <= theta < 1
        Dist.__init__(self, th=theta, eps=eps)

    def gen(self, x, th):
        return numpy.log((1-th*(1-x))/x)

    def igen(self, x, th):
        return (1-th)/(numpy.e**x-th)


class AliMikhailHaq(Copula):
    """
    Ali-Mikhail-Haq copula.

    Examples:
        >>> distribution = chaospy.AliMikhailHaq(
        ...     chaospy.Iid(chaospy.Uniform(), 2))
        >>> print(distribution)
        AliMikhailHaq(Iid(Uniform(lower=0, upper=1), 2), eps=1e-06, theta=2.0)
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        >>> print(numpy.around(distribution.sample(4), 4))
        >>> print(numpy.around(distribution.mom((1, 2)), 4))
    """

    def __init__(self, dist, theta=2., eps=1e-6):
        self._repr = {"theta": theta}
        trans = ali_mikhail_haq(theta, eps)
        return Copula.__init__(self, dist=dist, trans=trans)
