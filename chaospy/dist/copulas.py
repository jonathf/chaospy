"""
Collection of Copulas

To construct a copula one needs a copula transformation and the
Copula wrapper.

Examples
--------
>>> dist = cp.Iid(cp.Uniform(), 2)
>>> copula = cp.Gumbel(dist, theta=1.5)

The resulting copula is then ready for use
>>> cp.seed(1000)
>>> print copula.sample(5)
[[ 0.65358959  0.11500694  0.95028286  0.4821914   0.87247454]
 [ 0.02388273  0.10004972  0.00127477  0.10572619  0.4510529 ]]
"""
import numpy as np
import scipy as sp
from backend import Dist
from cores import student_t, mvstudentt


class Copula(Dist):

    def __init__(self, dist, trans):
        """
Parameters
----------
dist : Dist
    Distribution to wrap the copula around
trans : Dist
    The copula wrapper. [0,1]^D \into [0,1]^D
        """
        Dist.__init__(self, dist=dist, trans=trans,
                _advance=True, _length=len(trans))

    def _cdf(self, x, G):
        dist, trans = G.D["dist"], G.D["trans"]
        q = G(G(x, dist), trans)
        return q

    def _bnd(self, x, G):
        return G(x, G.D["dist"])

    def _ppf(self, q, G):
        dist, trans = G.D["dist"], G.D["trans"]
        return G(G(q, trans), dist)

    def _pdf(self, x, G):
        dist, trans = G.D["dist"], G.D["trans"]
        return G(G.fwd_as_pdf(x, dist), trans)*G(x, dist)


class Archimedia(Dist):
    """
Archimedean copula superclass. Subset this to generate an
archimedean.
    """

    def _ppf(self, x, th, eps):

        for i in xrange(1, len(x)):

            q = x[:i+1].copy()
            lo, up = 0,1
            dq = np.zeros(i+1)
            dq[i] = eps
            flo, fup = -q[i],1-q[i]

            for iteration in range(1, 10):
                fq = self._diff(q[:i+1], th, eps)
                dfq = self._diff((q[:i+1].T+dq).T, th, eps)
                dfq = (dfq-fq)/eps
                dfq = np.where(dfq==0, np.inf, dfq)

                fq = fq-x[i]
                if not np.any(np.abs(fq)>eps):
                    break

                # reduce boundaries
                flo = np.where(fq<=0, fq, flo)
                lo = np.where(fq<=0, q[i], lo)

                fup = np.where(fq>=0, fq, fup)
                up = np.where(fq>=0, q[i], up)

                # Newton increment
                qdq = q[i]-fq/dfq

                # if new val on interior use Newton
                # else binary search
                q[i] = np.where((qdq<up)*(qdq>lo),
                        qdq, .5*(up+lo))

            x[i] = q[i]
        return x


    def _cdf(self, x, th, eps):
        out = np.zeros(x.shape)
        out[0] = x[0]
        for i in xrange(1,len(x)):
            out[i][x[i]==1] = 1
            out[i] = self._diff(x[:i+1], th, eps)

        return out

    def _pdf(self, x, th, eps):
        out = np.ones(x.shape)
        sign = 1-2*(x>.5)
        for i in xrange(1,len(x)):
            x[i] += eps*sign[i]
            out[i] = self._diff(x[:i+1], th, eps)
            x[i] -= eps*sign[i]
            out[i] -= self._diff(x[:i+1], th, eps)
            out[i] /= eps

        out = abs(out)
        return out

    def _diff(self, x, th, eps):
        """
Numerical approximation of a Rosenblatt transformation created from
copula formulation.
        """
        foo = lambda y: self.igen(np.sum(self.gen(y, th), 0), th)

        out1 = out2 = 0.
        sign = 1 - 2*(x>.5).T
        for I in np.ndindex(*((2,)*(len(x)-1)+(1,))):

            eps_ = np.array(I)*eps
            x_ = (x.T + sign*eps_).T
            out1 += (-1)**sum(I)*foo(x_)

            x_[-1] = 1
            out2 += (-1)**sum(I)*foo(x_)

        out = out1/out2
        return out


    def _bnd(self, **prm):
        return 0,1


class gumbel(Archimedia):
    "Gumbel copula backend"

    def __init__(self, N, theta=1., eps=1e-6):
        theta = float(theta)
        Dist.__init__(self, th=theta, eps=eps, _length=N)
    def gen(self, x, th):
        return (-np.log(x))**th
    def igen(self, x, th):
        return np.e**(-x**th)


def Gumbel(dist, theta=2., eps=1e-6):
    r"""Gumbel Copula

Definition
----------
:math:
\phi(x;th) = \frac{x^{-th}-1}{th}
\phi^{-1}(q;th) = (q*th + 1)^{-1/th}

th : theta \in [1,inf)

Parameters
----------
dist : Dist
    The Distribution to wrap
theta : float
    Copula parameter

Returns
-------
out : Dist
    The resulting copula distribution.

Examples
--------
>>> dist = cp.J(cp.Uniform(), cp.Normal())
>>> copula = cp.Gumbel(dist, theta=2)
>>> print copula.sample(3, "S")
[[ 0.5         0.75        0.25      ]
 [ 0.07686128 -1.50814454  1.65112325]]
"""
    return Copula(dist, gumbel(len(dist), theta, eps))


class clayton(Archimedia):
    "clayton copula backend"

    def __init__(self, N, theta=1., eps=1e-6):
        Dist.__init__(self, th=float(theta), _length=N, eps=eps)
    def gen(self, x, th):
        return (x**-th-1)/th
    def igen(self, x, th):
        return (1.+th*x)**(-1./th)

def Clayton(dist, theta=2., eps=1e-6):
    return Copula(dist, clayton(len(dist), theta, eps))


class ali_mikhail_haq(Archimedia):
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

class frank(Archimedia):
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

class joe(Archimedia):
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

    def __init__(self, R):
        "R symmetric & positive definite matrix"
        C = np.linalg.cholesky(R)
        Ci = np.linalg.inv(C)
        Dist.__init__(self, C=C, Ci=Ci, _length=len(C))

    def _cdf(self, x, C, Ci):
        return sp.special.ndtr(np.dot(sp.special.ndtri(x).T, Ci)).T

    def _ppf(self, q, C, Ci):
        return sp.special.ndtr(np.dot(sp.special.ndtri(q).T, C)).T

    def _bnd(self, C, Ci):
        return 0.,1.

def Nataf(dist, R):
    "Nataf (normal) copula"
    return Copula(dist, nataf(R))


class t_copula(Dist):

    def __init__(self, a, R):
        self.MV = mvstudentt(a, np.zeros(len(R)), R)
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



if __name__=="__main__":
    import __init__ as cp
    import numpy as np
    import doctest
    doctest.testmod()
