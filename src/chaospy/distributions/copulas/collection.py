import numpy
from scipy import special

from .baseclass import Archimedean, Copula
from ..baseclass import Dist


class gumbel(Archimedean):
    "Gumbel copula backend"

    def __init__(self, N, theta=1., eps=1e-6):
        theta = float(theta)
        Dist.__init__(self, th=theta, eps=eps, _length=N)
    def gen(self, x, th):
        return (-numpy.log(x))**th
    def igen(self, x, th):
        return numpy.e**(-x**th)


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
    >>> dist = chaospy.J(chaospy.Uniform(), chaospy.Normal())
    >>> copula = chaospy.Gumbel(dist, theta=2)
    >>> print(copula.sample(3, "H"))
    [[ 0.125       0.625       0.375     ]
     [ 0.62638094 -0.45417192 -0.21628863]]
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
        return numpy.log((1-th*(1-x))/x)
    def igen(self, x, th):
        return (1-th)/(numpy.e**x-th)


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
        return -numpy.log((numpy.e**(-th*x)-1)/(numpy.e**-th-1))
    def igen(self, q, th):
        return -numpy.log(1+numpy.e**-q*(numpy.e**-th-1))/th

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
        return -numpy.log(1-(1-x)**th)

    def igen(self, q, th):
        return 1-(1-numpy.e**-q)**(1/th)

def Joe(dist, theta=2., eps=1e-6):
    "Joe copula"
    return Copula(dist, joe(len(dist), theta, eps))

class nataf(Dist):
    "Nataf (normal) copula backend"

    def __init__(self, R, ordering=None):
        "R symmetric & positive definite matrix"

        if ordering is None:
            ordering = range(len(R))
        ordering = numpy.array(ordering)

        P = numpy.eye(len(R))[ordering]

        R = numpy.dot(P, numpy.dot(R, P.T))
        R = numpy.linalg.cholesky(R)
        R = numpy.dot(P.T, numpy.dot(R, P))
        Ci = numpy.linalg.inv(R)
        Dist.__init__(self, C=R, Ci=Ci, _length=len(R))

    def _cdf(self, x, C, Ci):
        out = special.ndtr(numpy.dot(Ci, special.ndtri(x)))
        return out

    def _ppf(self, q, C, Ci):
        out = special.ndtr(numpy.dot(C, special.ndtri(q)))
        return out

    def _bnd(self, C, Ci):
        return 0.,1.

def Nataf(dist, R, ordering=None):
    "Nataf (normal) copula"
    return Copula(dist, nataf(R, ordering))


class t_copula(Dist):

    def __init__(self, a, R):
        from ..cores import mvstudentt, student_t
        self.MV = mvstudentt(a, numpy.zeros(len(R)), R)
        self.UV = student_t(a)
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
