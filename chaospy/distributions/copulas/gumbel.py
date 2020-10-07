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

from ..baseclass import CopulaDistribution
from .archimedean import Archimedean


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


class Gumbel(CopulaDistribution):
    r"""
    Gumbel Copula.

    .. math::
        \phi(x;th) = \frac{x^{-th}-1}{th}
        \phi^{-1}(q;th) = (q*th + 1)^{-1/th}

    where `th` (or theta) is defined on the interval `[1,inf)`.

    Args:
        dist (Distribution):
            The distribution to wrap
        theta (float):
            Copula parameter

    Examples:
        >>> distribution = chaospy.Gumbel(
        ...     chaospy.Iid(chaospy.Uniform(-1, 1), 2), theta=2)
        >>> distribution
        Gumbel(Iid(Uniform(lower=-1, upper=1), 2), theta=2)
        >>> samples = distribution.sample(3)
        >>> samples.round(4)
        array([[ 0.3072, -0.77  ,  0.9006],
               [ 0.4709,  0.2102,  0.8696]])
        >>> distribution.pdf(samples).round(4)
        array([7.9975000e+00, 5.0700000e-02, 3.6718099e+03])
        >>> distribution.fwd(samples).round(4)
        array([[0.6536, 0.115 , 0.9503],
               [0.4822, 0.8725, 0.2123]])
        >>> mesh = numpy.meshgrid([.4, .5, .6], [.4, .5, .6])
        >>> distribution.inv(mesh).round(4)
        array([[[-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ],
                [-0.2   ,  0.    ,  0.2   ]],
        <BLANKLINE>
               [[-0.0022,  0.1573,  0.3174],
                [ 0.109 ,  0.2591,  0.4062],
                [ 0.2181,  0.3564,  0.489 ]]])

    """

    def __init__(self, dist, theta, eps=1e-6):
        super(Gumbel, self).__init__(
            dist=dist,
            trans=gumbel(len(dist), theta),
            repr_args=[dist, "theta=%s" % theta],
        )
