r"""
Clayton Copula.

The calculation of the derivative of the ``iphi`` function:

.. math::
  \begin{eqnarray}
    iphi(u)     &= (1+\theta*u)^(-1/\theta) \\
    iphi'(u)    &= \theta*(-1/\theta)*(1+\theta*u)^(-1/\theta-1)
                = \theta*\sigma(1+\theta*u, 1, \theta) \\
    iphi''(u)   &= \theta*(-1/\theta)*theta*(-1/\theta-1)*(1+\theta*u)^{-1/\theta-2}
                = \theta^2*\sigma(1+\theta*u, 2, \theta) \\
                & \dots \\
    iphi^(n)(u) &= \theta^n*\prod_{d=0}^{n-1}(-1/\theta-d)*(1+\theta*u)^{-1/\theta-n}
                = \theta^n*sigma(1+theta*u, n)
  \end{eqnarray}
"""
import numpy
from scipy import special

from .baseclass import Copula
from .archimedean import Archimedean
from ..baseclass import Dist


class clayton(Archimedean):
    """Clayton copula."""

    def _phi(self, t_loc, theta):
        return (t_loc**-theta-1)/theta

    def _delta_phi(self, t_loc, theta):
        return -t_loc**(-theta-1)

    def _inverse_phi(self, u_loc, theta, order):
        return theta**order*self._sigma(1+theta*u_loc, theta, order)


class Clayton(Copula):
    """
    Clayton Copula.

    Args:
        dist (Dist):
            The Distribution to wrap
        theta (float):
            Copula parameter. Required to be above 0.

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
         [1.9803 1.4482 1.5536]
         [1.0651 1.1643 1.686 ]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[0.6017 0.3102 0.6819 0.209 ]
         [0.631  0.4154 0.625  0.125 ]]
        >>> print(numpy.around(distribution.mom((1, 2)), 4))
        0.2196
    """

    def __init__(self, dist, theta=2.):
        """
        Args:
            dist (Dist):
                The Distribution to wrap
            theta (float):
                Copula parameter
        """
        assert theta > 0
        self._repr = {"theta": theta}
        trans = clayton(len(dist), theta=theta)
        return Copula.__init__(self, dist=dist, trans=trans)
