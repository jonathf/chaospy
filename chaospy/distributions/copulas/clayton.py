"""Clayton copula."""
import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist

class clayton(Archimedean):
    """Clayton copula."""

    def __init__(self, length, theta=1., eps=1e-6):
        self.length = length
        Dist.__init__(self, th=float(theta), eps=eps)

    def __len__(self):
        return self.length

    def gen(self, x, th):
        return (x**-th-1)/th

    def igen(self, x, th):
        return (1.+th*x)**(-1./th)


class Clayton(Copula):
    """
    Clayton Copula.

    Args:
        dist (Dist) : The Distribution to wrap
        theta (float) : Copula parameter

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
         [1.9803 1.4482 1.5538]
         [1.0651 1.1642 1.6861]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[0.6536 0.115  0.9503 0.4822]
         [0.9043 0.0852 0.3288 0.4633]]
        >>> print(numpy.around(distribution.mom((1, 2)), 4))
        0.2196
    """

    def __init__(self, dist, theta=2., eps=1e-6):
        """
        Args:
            dist (Dist) : The Distribution to wrap
            theta (float) : Copula parameter
        """
        self._repr = {"theta": theta}
        trans = clayton(len(dist), theta=theta, eps=eps)
        return Copula.__init__(self, dist=dist, trans=trans)


class Clayton_(Dist):
    """
    Examples:
        >>> distribution = Clayton_(2, theta=2)
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)]*2)
        >>> print(numpy.array(mesh))
        [[[0.   0.25 0.5  0.75 1.  ]
          [0.   0.25 0.5  0.75 1.  ]
          [0.   0.25 0.5  0.75 1.  ]
          [0.   0.25 0.5  0.75 1.  ]
          [0.   0.25 0.5  0.75 1.  ]]
        <BLANKLINE>
         [[0.   0.   0.   0.   0.  ]
          [0.25 0.25 0.25 0.25 0.25]
          [0.5  0.5  0.5  0.5  0.5 ]
          [0.75 0.75 0.75 0.75 0.75]
          [1.   1.   1.   1.   1.  ]]]
        >>> print(numpy.around(distribution.fwd(mesh), 4))
        [[[0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]]
        <BLANKLINE>
         [[0.     0.     0.     0.     0.    ]
          [0.25   0.3708 0.0966 0.0345 0.0156]
          [0.5    0.7728 0.432  0.227  0.125 ]
          [0.75   0.9313 0.766  0.5802 0.4219]
          [1.     1.     1.     1.     1.    ]]]
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]]
        <BLANKLINE>
         [[1.     0.     0.     0.     0.    ]
          [1.     0.3708 0.0966 0.0345 0.0156]
          [1.     0.7728 0.432  0.227  0.125 ]
          [1.     0.9313 0.766  0.5802 0.4219]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]
          [0.     0.25   0.5    0.75   1.    ]]
        <BLANKLINE>
         [[0.     0.     0.     0.     0.    ]
          [0.25   0.25   0.25   0.227  0.125 ]
          [0.5    0.7728 0.5    0.5    0.4219]
          [0.75   0.7728 0.75   0.75   0.7443]
          [1.     1.     1.     1.     1.    ]]]
    """

    def __init__(self, length, theta=1.):
        self.length = length
        Dist.__init__(self, theta=float(theta))

    def __len__(self):
        return self.length

    def _lower(self, theta):
        return 0.

    def _upper(self, theta):
        return 1.

    def _cdf(self, x, theta):
        return cdf(x, theta)

    def _pdf(self, x, theta):
        return pdf(x, theta)


def phi(t_loc, theta):  # confirmed
    out = (t_loc**-theta-1)/theta
    return out

def dphi(t_loc, theta):
    out = -t_loc**(-theta-1)
    return out


def S(u_loc, theta, order):
    out = 1.
    for dim in range(order):
        out *= (-1/theta-dim)
    assert numpy.all(u_loc > 0), u_loc
    out = out*u_loc**(-1/theta-order)
    return out

def iphi(u_loc, theta, order):  # confirmed order 0
    out = theta**order*S(1+theta*u_loc, theta, order)
    if not order:
        out = numpy.clip(out, 0, 1)
    return out

def C(x_loc, theta, order=0):  # confirmed order 0
    assert numpy.all(x_loc <= 1)
    assert numpy.all(x_loc >= 0)
    out = iphi(numpy.sum(phi(x_loc, theta), 0), theta, order)
    if order:
        out *= numpy.where(out, numpy.prod(dphi(x_loc[:order], theta), 0), 0)
    return out


def pdf_(x_loc, theta, order=0):
    loc = numpy.ones(x_loc.shape)
    loc[:order] = x_loc[:order]
    out = C(loc, theta, order=order)
    loc[order] = x_loc[order]
    index = out != 0
    out[index] = C(loc, theta, order=order+1)[index]/out[index]
    out[~index] = 0
    return out


def pdf(x_loc, theta):
    return numpy.vstack([
        pdf_(x_loc, theta, order)
        for order in range(len(x_loc))
    ])


def cdf_(x_loc, theta, order=0):
    assert numpy.all(x_loc <= 1)
    assert numpy.all(x_loc >= 0)
    loc = numpy.ones(x_loc.shape)
    loc[:order] = x_loc[:order]
    out = C(loc, theta, order=order)
    loc[order] = x_loc[order]
    index = out != 0
    out[index] = C(loc, theta, order=order)[index]/out[index]
    out[~index] = 1
    return out


def cdf(x_loc, theta):
    assert numpy.all(x_loc <= 1)
    assert numpy.all(x_loc >= 0)
    out = numpy.vstack([
        cdf_(x_loc, theta, order)
        for order in range(len(x_loc))
    ])
    return out
