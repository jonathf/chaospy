r"""
Gumbel Copula.

As a short hand:

.. math::

    s_n = \sigma(u, n, \theta)

The calculation of the derivative of the ``iphi`` function:

.. math::
  \begin{eqnarray}
    iphi(u)     &= e^{-u^{1/\theta}}
    iphi'(u)    &= e^{-u^{1/theta}}*(-1/theta*u^{1/theta-1})
                = iphi(u)*s_1 \\
    iphi''(u)   &= iphi'(u)*s_1+iphi(u)*s_2
                = iphi(u)*s_1^2+iphi(u)*s_2
                = iphi(u)*(s_1^2+s_2) \\
    iphi'''(u)  &= iphi'(u)*(s_1^2+s_2)+iphi(u)*(2*s_1*s_2+s_3)
                = iphi(u)*s_1*(s_1^2+s_2)+iphi(u)*(2*s_1*s_2+s_3)
                = iphi(u)*(s_1^3+3*s_1*s_2+s_3) \\
    iphi''''(u) &= iphi'(u)*(s_1^3+3*s_1*s_2+s_3) +
                   iphi(u)*(3*s_1^2*s_2+3*s_2^2+3*s_1*s_3+s_4)
                = iphi(u)*s_1*(s_1^3+3*s_1*s_2+s_3) +
                  iphi(u)*(3*s_1^2*s_2+3*s_2^2+3*s_1*s_3+s_4)
                = iphi(u)*(s_1^4 + 3*s_1^2*s_2 + s_1*s_3 +
                           3*s_1^2*s_2 + 3*s_2^2 + 3*s_1*s_3 + s_4)
                = iphi(u)*(s_1^4 + 6*s_1^2*s_2 +
                           s_1*s_3 + 3*s_2^2 + 3*s_1*s_3 + s_4)
  \end{eqnarray}

The same formula defined recursively:

.. math::
  \begin{eqnarray}
    iphi'(u)    &= iphi(u)*\sigma(u, 1, \theta) \\
    iphi''(u)   &= d/du ( iphi(u)*\sigma(u, 1, \theta) )
                = iphi'(u)*\sigma(u, 1, \theta) +
                  iphi(u)*\sigma(u, 2, \theta) \\
    iphi'''(u)  &= d/du (iphi'(u)*\sigma(u, 1, \theta) +
                          iphi(u)*\sigma(u, 2, \theta))
                = iphi''(u)*\sigma(u, 1, \theta) +
                  iphi'(u)*\sigma(u, 2, \theta) +
                  iphi'(u)*\sigma(u, 2, \theta) +
                  iphi(u)*\sigma(u, 3, \theta)
                = iphi''(u)*\sigma(u, 1, \theta) +
                  2*iphi'(u)*\sigma(u, 2, \theta) +
                  iphi(u)*\sigma(u, 3, \theta) \\
    iphi''''(u) &= d/du (iphi''(u)*\sigma(u, 1, \theta) +
                         2*iphi'(u)*\sigma(u, 2, \theta) +
                         iphi(u)*\sigma(u, 3, \theta))
                = iphi'''(u)*\sigma(u, 1, \theta) +
                  iphi''(u)*\sigma(u, 2, \theta) +
                  2*iphi''(u)*\sigma(u, 2, \theta) +
                  2*iphi'(u)*\sigma(u, 3, \theta) +
                  iphi'(u)*\sigma(u, 3, \theta) +
                  iphi(u)*\sigma(u, 4, \theta)
                = iphi'''(u)*\sigma(u, 1, \theta) +
                  3*iphi''(u)*\sigma(u, 2, \theta) +
                  3*iphi'(u)*\sigma(u, 3, \theta) +
                  iphi(u)*\sigma(u, 4, \theta) \\
                & \vdots \\
    iphi^{n}(u) &= sum_{i=1}^{n} comb(n, i-1) iphi^{n-i}(u) \sigma(u, i, \theta)
  \end{eqnarray}
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


class gumbel(Archimedean):
    """Gumbel backend."""

    def _phi(self, t_loc, theta):
        return (-numpy.log(t_loc))**theta

    def _delta_phi(self, t_loc, theta):
        return -theta*(-numpy.log(t_loc))**(theta-1)/t_loc

    def _inverse_phi(self, u_loc, theta, order):
        @lru_cache(None)
        def iphi(n):
            if n:
                return sum(special.comb(n, i-1)*iphi(n-i)*sigma(i)
                           for i in range(1, n+1))
            return numpy.e**(-u_loc**(1/theta))
        @lru_cache(None)
        def sigma(n):
            return self._sigma(u_loc, theta, n)
        return iphi(order)


class Gumbel(Copula):
    r"""
    Gumbel Copula.

    .. math::
        \phi(x;th) = \frac{x^{-th}-1}{th}
        \phi^{-1}(q;th) = (q*th + 1)^{-1/th}

    where `th` (or theta) is defined on the interval `[1,inf)`.

    Args:
        dist (Dist):
            The Distribution to wrap
        theta (float):
            Copula parameter

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
         [[0.2843 0.4898 0.7218]
          [0.4348 0.6296 0.8129]
          [0.5968 0.7532 0.882 ]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[  1.0732   5.3662  59.9037]
         [  1.2609   7.9291 108.0494]
         [  1.0296   7.6845 120.2633]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[0.4868 0.2788 0.5216 0.4511]
         [0.5322 0.061  0.7691 0.8187]]
    """

    def __init__(self, dist, theta, eps=1e-6):
        """
        Args:
            dist (Dist) : The Distribution to wrap
            theta (float) : Copula parameter
        """
        self._repr = {"theta": theta}
        Copula.__init__(self, dist=dist, trans=gumbel(len(dist), theta))
