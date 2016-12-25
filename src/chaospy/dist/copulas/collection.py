import numpy as np
import scipy as sp

import chaospy.dist

from chaospy.dist.baseclass import Dist
from chaospy.dist.copulas.baseclass import Archimedean, Copula

class gumbel(Archimedean):
    "Gumbel copula backend"

    def __init__(self, N, theta=1., eps=1e-6):
        theta = float(theta)
        Dist.__init__(self, th=theta, eps=eps, _length=N)
    def gen(self, x, th):
        return (-np.log(x))**th
    def igen(self, x, th):
        return np.e**(-x**th)


def Gumbel(dist, theta=2., eps=1e-6):
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
    >>> dist = cp.J(cp.Uniform(), cp.Normal())
    >>> copula = cp.Gumbel(dist, theta=2)
    >>> print(copula.sample(3, "S"))
    [[ 0.5         0.75        0.25      ]
    [ 0.07686128 -1.50814454  1.65112325]]
"""
    return Copula(dist, gumbel(len(dist), theta, eps))


class clayton(Archimedean):
    "clayton copula backend"

    def __init__(self, N, theta=1., eps=1e-6):
        Dist.__init__(self, th=float(theta), _length=N, eps=eps)
    def gen(self, x, th):
        return (x**-th-1)/th
    def igen(self, x, th):
        return (1.+th*x)**(-1./th)

def Clayton(dist, theta=2., eps=1e-6):
    return Copula(dist, clayton(len(dist), theta, eps))


class ali_mikhail_haq(Archimedean):
    "Ali Mikhail Haq copula backend"

    def __init__(self, N, theta=.5, eps=1e-6):
        theta = float(theta)
        assert -1<=theta<1
        Dist.__init__(self, th=theta, _length=N, eps=eps)
    def gen(self, x, th):
        return np.log((1-th*(1-x))/x)
    def igen(self, x, th):
        return (1-th)/(np.e**x-th)


def Ali_mikhail_haq(dist, theta=2., eps=1e-6):
    "Ali Mikhail Haq copula"
    trans = ali_mikhail_haq(len(dist), theta, eps)
    return Copula(dist, trans)

class frank(Archimedean):
    "Frank copula backend"

    def __init__(self, N, theta, eps=1e-6):
        "theta!=0"
        theta = float(theta)
        assert theta!=0
        Dist.__init__(self, th=theta, _length=N, eps=eps)

    def gen(self, x, th):
        return -np.log((np.e**(-th*x)-1)/(np.e**-th-1))
    def igen(self, q, th):
        return -np.log(1+np.e**-q*(np.e**-th-1))/th

def Frank(dist, theta=1., eps=1e-4):
    "Frank copula"
    return Copula(dist, frank(len(dist), theta, eps))

class joe(Archimedean):
    "Joe copula backend"

    def __init__(self, N, theta, eps=1e-6):
        "theta in [1,inf)"
        theta = float(theta)
        assert theta>=1
        Dist.__init__(self, th=theta, _length=N, eps=eps)

    def gen(self, x, th):
        return -np.log(1-(1-x)**th)

    def igen(self, q, th):
        return 1-(1-np.e**-q)**(1/th)

def Joe(dist, theta=2., eps=1e-6):
    "Joe copula"
    return Copula(dist, joe(len(dist), theta, eps))

class nataf(Dist):
    "Nataf (normal) copula backend"

    def __init__(self, R, ordering=None):
        "R symmetric & positive definite matrix"

        if ordering is None:
            ordering = range(len(R))
        ordering = np.array(ordering)

        P = np.eye(len(R))[ordering]

        R = np.dot(P, np.dot(R, P.T))
        R = np.linalg.cholesky(R)
        R = np.dot(P.T, np.dot(R, P))
        Ci = np.linalg.inv(R)
        Dist.__init__(self, C=R, Ci=Ci, _length=len(R))

    def _cdf(self, x, C, Ci):
        out = sp.special.ndtr(np.dot(Ci, sp.special.ndtri(x)))
        return out

    def _ppf(self, q, C, Ci):
        out = sp.special.ndtr(np.dot(C, sp.special.ndtri(q)))
        return out

    def _bnd(self, C, Ci):
        return 0.,1.

def Nataf(dist, R, ordering=None):
    "Nataf (normal) copula"
    return Copula(dist, nataf(R, ordering))


class t_copula(Dist):

    def __init__(self, a, R):
        self.MV = chaospy.dist.mvstudentt(a, np.zeros(len(R)), R)
        self.UV = chaospy.dist.student_t(a)
        Dist.__init__(self, _length=len(R))

    def _cdf(self, x):
        out = self.MV.fwd(self.UV.inv(x))
        return out

    def _ppf(self, q):
        out = self.MV.inv(q)
        out = self.UV.fwd(out)
        return out

    def _bnd(self):
        return 0.,1.

def T_copula(dist, a, R):
    return Copula(dist, t_copula(a, R))



if __name__=="__main__":
    import chaospy as cp
    import numpy as np
    import doctest
    doctest.testmod()
