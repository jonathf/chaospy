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

from ..baseclass import CopulaDistribution
from .archimedean import Archimedean


class clayton(Archimedean):
    """Clayton copula."""

    def _phi(self, t_loc, theta):
        return (t_loc**-theta-1)/theta

    def _delta_phi(self, t_loc, theta):
        return -t_loc**(-theta-1)

    def _inverse_phi(self, u_loc, theta, order):
        return theta**order*self._sigma(1+theta*u_loc, theta, order)


class Clayton(CopulaDistribution):
    """
    Clayton Copula.

    Examples:
        >>> distribution = chaospy.Clayton(
        ...     chaospy.Iid(chaospy.Uniform(-1, 1), 2), theta=2)
        >>> distribution
        Clayton(Iid(Uniform(lower=-1, upper=1), 2), theta=2)
        >>> samples = distribution.sample(3)
        >>> samples.round(4)
        array([[ 0.3072, -0.77  ,  0.9006],
               [ 0.2736, -0.3015,  0.1539]])
        >>> distribution.pdf(samples).round(4)
        array([0.3679, 0.1855, 0.2665])
        >>> distribution.fwd(samples).round(4)
        array([[0.6536, 0.115 , 0.9503],
               [0.4822, 0.8725, 0.2123]])
        >>> mesh = numpy.meshgrid([.4, .5, .6], [.4, .5, .6])
        >>> distribution.inv(mesh).round(4)
        array([[[-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ]],
        <BLANKLINE>
               [[-0.2008, -0.0431,  0.0945],
                [-0.0746,  0.0928,  0.2329],
                [ 0.0636,  0.2349,  0.3713]]])

    """

    def __init__(self, dist, theta=2.):
        """
        Args:
            dist (Distribution):
                The distribution to wrap
            theta (float):
                Copula parameter. Required to be above 0.
        """
        assert theta > 0
        return super(Clayton, self).__init__(
            dist=dist,
            trans=clayton(len(dist), theta=theta),
            repr_args=[dist, "theta=%s" % theta],
        )
