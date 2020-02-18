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

from .baseclass import Copula
from .archimedean import Archimedean
from ..baseclass import Dist


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


class Joe(Copula):
    """
    Joe Copula

    where `theta` is defined on the interval `[1,inf)`.
    """

    def __init__(self, dist, theta=2.):
        """
        Args:
            dist (Dist):
                The Distribution to wrap
            theta (float):
                Copula parameter
        """
        self._repr = {"theta": theta}
        Copula.__init__(self, dist, joe(len(dist), theta))
