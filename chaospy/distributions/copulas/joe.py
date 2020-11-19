r"""
Joe Copula.

The calculation of the derivative of the ``iphi`` function:

.. math::
  \begin{eqnarray}
    iphi(u)     &= 1-(1-e^{-u})^{1/\theta} \\
    iphi'(u)    &= -(1/theta)*(1-e^(-u))^(1/theta-1)*e^(-u)
                = -\sigma(1-e^-u, 1, \theta) = j1 \\
    iphi''(u)   &= d/du ( j1 )
                = j2 - j1
                = j2 - iphi'(u) \\
    iphi'''(u)  &= d/du ( j2 - j1 )
                = j3 - 2j2 - j2 + j1
                = (j3 - 2j2) - (j2 - j1)
                = (j3 - 2j2) - iphi''(u) \\
    iphi''''(u) &= d/du ( (j3 - 2j2) - (j2 - j1) )
                = ((j4 - 3j3) - 2(j3-2j2)) - ((j3-2j2) - (j2-j1))
                = ((j4 - 3j3) - 2(j3-2j2)) - iphi'''(u) \\
    iphi'''''(u)&= d/du ((j4 - 3j3) - 2(j3-2j2)) - ((j3-2j2) - (j2-j1))
                = ((j5-4j4) - 3(j4-3j3)) - 2((j4-3j3)-2(j3-2j2)) -
                   (((j4-3j3)-2(j3-2j2)) - ((j3-2j2)-(j2-j1)))
                = ((j5-4j4) - 3(j4-3j3)) - 2((j4-3j3)-2(j3-2j2)) - iphi''''(u) \\
  \end{eqnarray}

Here the notation :math:`jn` is a short hand and means:

.. math::
    jn = J(u, n) = -\sigma(1-e^-u, n) e^{-un}

which was the property:

.. math::

    d/du j(u, n) = J(u, n+1) - n*J(u, n)

This problem can be solved recursively using the function:

.. math::
  \begin{eqnarray}
    \rho(u, n, \theta, m) &= \sigma(1-e^{-u}, \theta, n) e^{-n\theta} & n &= m \\
    \rho(u, n, \theta, m) &= \rho(u, n, \theta, m+1) - m \rho(u, n-1, \theta, m) & n &\neq m \\
  \end{eqnarray}

Solution is then just:
.. math::
    iphi^{(n)}(u) = \rho(u, n, \theta, 1)
"""
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache
import numpy
from scipy import special

from ..baseclass import CopulaDistribution
from .archimedean import Archimedean


class joe(Archimedean):
    """Joe copula."""

    def _phi(self, t_loc, theta):
        return -numpy.log(1-(1-t_loc)**theta)

    def _delta_phi(self, t_loc, theta):
        return theta*(1-t_loc)**(theta-1)/(1-(1-t_loc)**theta)

    def _inverse_phi(self, u_loc, theta, order):
        if not order:
            return 1-(1-numpy.e**-u_loc)**(1/theta)
        @lru_cache(None)
        def rho(n, m=1):
            if n == m:
                return self._sigma(1-numpy.e**-u_loc, theta, n)*numpy.e**(-n*theta)
            return rho(n, m+1)-m*rho(n-1, m)
        return rho(order)


class Joe(CopulaDistribution):
    """
    Joe Copula.

    Examples:
        >>> distribution = chaospy.Joe(
        ...     chaospy.Iid(chaospy.Uniform(-1, 1), 2), theta=2)
        >>> distribution
        Joe(Iid(Uniform(lower=-1, upper=1), 2), theta=2)
        >>> samples = distribution.sample(3)
        >>> samples.round(4)
        array([[ 0.3072, -0.77  ,  0.9006],
               [ 0.4155, -0.173 ,  0.8661]])
        >>> distribution.pdf(samples).round(4)
        array([ 0.2014,  0.3844, 11.8495])
        >>> distribution.fwd(samples).round(4)
        array([[0.6536, 0.115 , 0.9503],
               [0.4822, 0.8725, 0.2123]])
        >>> mesh = numpy.meshgrid([.4, .5, .6], [.4, .5, .6])
        >>> distribution.inv(mesh).round(4)
        array([[[-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ]],
        <BLANKLINE>
               [[-0.3764, -0.0596,  0.199 ],
                [-0.1496,  0.115 ,  0.331 ],
                [ 0.0446,  0.2645,  0.444 ]]])

    """

    def __init__(self, dist, theta=2.):
        """
        Args:
            dist (Distribution):
                The distribution to wrap
            theta (float):
                Copula parameter. Required to be above 1.
        """
        super(Joe, self).__init__(
            dist=dist,
            trans=joe(len(dist), theta),
            repr_args=[dist, "theta=%s" % theta],
        )
