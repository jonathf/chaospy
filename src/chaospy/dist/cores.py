"""
Backend for the collection distributions.

To create a user-defined distribution use the distributions in
this module as template.

Documentation for each distribution is available in
dist.collection.
"""
import numpy as np
import scipy as sp
from scipy import special

from chaospy.dist.baseclass import Dist

import chaospy.dist


class uniform(Dist):

    def __init__(self):
        Dist.__init__(self)
    def _pdf(self, x):
        return 0.5
    def _cdf(self, x):
        return .5*x+.5
    def _ppf(self, q):
        return 2*q-1
    def _bnd(self):
        return -1.,1.
    def _mom(self, k):
        return 1./(k+1)*(k%2==0)
    def _ttr(self, n):
        return 0., n*n/(4.*n*n-1)
    def _str(self):
        return "uni"

class loguniform(Dist):

    def __init__(self, lo=0, up=1):
        Dist.__init__(self, lo=lo, up=up)
    def _pdf(self, x, lo, up):
        return 1./(x*(up-lo))
    def _cdf(self, x, lo, up):
        return (np.log(x)-lo)/(up-lo)
    def _ppf(self, q, lo, up):
        return np.e**(q*(up-lo) + lo)
    def _bnd(self, lo, up):
        return np.e**lo, np.e**up
    def _mom(self, k, lo, up):
        return ((np.e**(up*k)-np.e**(lo*k))/((up-lo)*(k+(k==0))))**(k!=0)
    def _str(self, lo, up):
        return "loguni(%s,%s)" % (lo, up)

class normal(Dist):

    def __init__(self):
        Dist.__init__(self)
    def _pdf(self, x):
        return (2*np.pi)**(-.5)*np.e**(-x**2/2.)
    def _cdf(self, x):
        return special.ndtr(x)
    def _ppf(self, x):
        return special.ndtri(x)
    def _mom(self, k):
        return .5*sp.misc.factorial2(k-1)*(1+(-1)**k)
    def _ttr(self, n):
        return 0., 1.*n
    def _bnd(self):
        return -7.5, 7.5
    def _str(self):
        return "nor"

class lognormal(Dist):

    def __init__(self, a=1):
        Dist.__init__(self, a=a)
    def _pdf(self, x, a):
        out = np.e**(-np.log(x+(1-x)*(x<=0))**2/(2*a*a)) / \
            ((x+(1-x)*(x<=0))*a*np.sqrt(2*np.pi))*(x>0)
        return out
    def _cdf(self, x, a):
        return special.ndtr(np.log(x+(1-x)*(x<=0))/a)*(x>0)
    def _ppf(self, x, a):
        return np.e**(a*special.ndtri(x))
    def _mom(self, k, a):
        return np.e**(.5*a*a*k*k)
    def _ttr(self, n, a):
        return \
    (np.e**(n*a*a)*(np.e**(a*a)+1)-1)*np.e**(.5*(2*n-1)*a*a), \
                (np.e**(n*a*a)-1)*np.e**((3*n-2)*a*a)
    def _bnd(self, a):
        return 0, self._ppf(1-1e-10, a)
    def _str(self, a):
        return "lognor(%s)" % a


class expon(Dist):

    def __init__(self):
        Dist.__init__(self)
    def _pdf(self, x):
        return np.e**-x
    def _cdf(self, x):
        return 1.-np.e**-x
    def _ppf(self, q):
        return -np.log(1-q)
    def _mom(self, k):
        return sp.misc.factorial(k)
    def _ttr(self, n):
        return 2*n+1, n*n
    def _bnd(self):
        return 0, 42.
    def _str(self):
        return "expon"

class gamma(Dist):

    def __init__(self, a=1):
        Dist.__init__(self, a=a)
    def _pdf(self, x, a):
        return x**(a-1)*np.e**(-x) / special.gamma(a)
    def _cdf(self, x, a):
        return special.gammainc(a, x)
    def _ppf(self, q, a):
        return special.gammaincinv(a, q)
    def _mom(self, k, a):
        return special.gamma(a+k)/special.gamma(a)
    def _ttr(self, n, a):
        return 2*n+a, n*n+n*(a-1)
    def _bnd(self, a):
        return 0, 40+2*a
    def _str(self, a):
        return "gam(%s)" % a

class laplace(Dist):

    def __init__(self):
        Dist.__init__(self)
    def _pdf(self, x):
        return np.e**-np.abs(x)/2
    def _cdf(self, x):
        return (1+np.sign(x)*(1-np.e**-abs(x)))/2
    def _mom(self, k):
        return .5*sp.misc.factorial(k)*(1+(-1)**k)
    def _ppf(self, x):
        return np.where(x>.5, -np.log(2*(1-x)), np.log(2*x))
    def _bnd(self):
        return -32., 32.
    def _str(self):
        return "lap"

class beta(Dist):

    def __init__(self, a=1, b=1):
        Dist.__init__(self, a=a, b=b)
    def _pdf(self, x, a, b):
        return x**(a-1)*(1-x)**(b-1)/ \
            special.beta(a, b)
    def _cdf(self, x, a, b):
        return special.btdtr(a, b, x)
    def _ppf(self, q, a, b):
        return special.btdtri(a, b, q)
    def _mom(self, k, a, b):
        return special.beta(a+k,b)/special.beta(a,b)
    def _ttr(self, n, a, b):

        nab = 2*n+a+b
        A = ((a-1)**2-(b-1)**2)*.5/\
                (nab*(nab-2) + (nab==0) + (nab==2)) + .5
        B1 = a*b*1./((a+b+1)*(a+b)**2)
        B2 = (n+a-1)*(n+b-1)*n*(n+a+b-2.)/\
            ((nab-1)*(nab-3)*(nab-2)**2+2.*((n==0)+(n==1)))
        B = np.where((n==0)+(n==1), B1, B2)
        return A, B
    def _bnd(self, a, b):
        return 0., 1.
    def _str(self, a, b):
        return "bet(%s,%s)" % (a,b)


class weibull(Dist):

    def __init__(self, a=1):
        Dist.__init__(self, a=a)
    def _pdf(self, x, a):
        return a*x**(a-1)*np.e**(-x**a)
    def _cdf(self, x, a):
        return (1-np.e**(-x**a))
    def _ppf(self, q, a):
        return (-np.log(1-q+1*(q==1)))**(1./a)*(q!=1) +\
            30.**(1./a)*(q==1)
    def _mom(self, k, a):
        return special.gamma(1.+k*1./a)
    def _bnd(self, a):
        return 0, 30.**(1./a)
    def _str(self, a):
        return "wei(%s)" % a

def tri_ttr(k, a):
    from chaospy.quadrature import clenshaw_curtis
    q1,w1 = clenshaw_curtis(int(10**3*a), 0, a)
    q2,w2 = clenshaw_curtis(int(10**3*(1-a)), a, 1)
    q = np.concatenate([q1,q2], 1)
    w = np.concatenate([w1,w2])
    w = w*np.where(q<a, 2*q/a, 2*(1-q)/(1-a))

    from chaospy.poly import variable
    x = variable()

    orth = [x*0, x**0]
    inner = np.sum(q*w, -1)
    norms = [1., 1.]
    A,B = [],[]

    for n in range(k):
        A.append(inner/norms[-1])
        B.append(norms[-1]/norms[-2])
        orth.append((x-A[-1])*orth[-1]-orth[-2]*B[-1])

        y = orth[-1](*q)**2*w
        inner = np.sum(q*y, -1)
        norms.append(np.sum(y, -1))

    A, B = np.array(A).T[0], np.array(B).T
    return A, B


class triangle(Dist):

    def __init__(self, a=.5):
        assert np.all(a>=0) and np.all(a<=1)
        Dist.__init__(self, a=a)
    def _pdf(self, D, a):
        return np.where(D<a, 2*D/a, 2*(1-D)/(1-a))
    def _cdf(self, D, a):
        return np.where(D<a, D**2/(a + (a==0)),
                (2*D-D*D-a)/(1-a+(a==1)))
    def _ppf(self, q, a):
        return np.where(q<a, np.sqrt(q*a), 1-np.sqrt(1-a-q*(1-a)))
    def _mom(self, k, a):
        a_ = a*(a!=1)
        out = 2*(1.-a_**(k+1))/((k+1)*(k+2)*(1-a_))
        return np.where(a==1, 2./(k+2), out)
    def _bnd(self, a):
        return 0., 1.
    def _str(self, a):
        return "tri(%s)" % a
    def _ttr(self, k, a):
        a = a.item()
        if a==0: return beta()._ttr(k, 1, 2)
        if a==1: return beta()._ttr(k, 2, 1)

        A,B = tri_ttr(np.max(k)+1, a)
        A = np.array([[A[_] for _ in k[0]]])
        B = np.array([[B[_] for _ in k[0]]])
        return A,B


#  class wigner(Dist):
#
#      def __init__(self):
#          Dist.__init__(self)
#
#      def _pdf(self, x):
#          return 2*np.sqrt(1-x*x)/np.pi
#      def _cdf(self, x):
#          return special.btdtr(1.5, 1.5, .5*x+.5)
#      def _ppf(self, q):
#          return 2*special.btdtri(1.5, 1.5, q)-1
#      def _mom(self, n):
#          return ((n+1)%2)*comb(n, n/2)/((.5*n+1)*2**n)
#      def _bnd(self):
#          return -1.,1.
#      def _ttr(self, n):
#          return 0., .25**(n!=0)
#      def _str(self):
#          return "w"


class kumaraswamy(Dist):

    def __init__(self, a=1, b=1):
        assert np.all(a>0) and np.all(b>0)
        Dist.__init__(self, a=a, b=b)

    def _pdf(self, x, a, b):
        return a*b*x**(a-1)*(1-x**a)**(b-1)

    def _cdf(self, x, a, b):
        return 1-(1-x**a)**b

    def _ppf(self, q, a, b):
        return (1-(1-q)**(1./b))**(1./a)

    def _mom(self, k, a, b):
        return b*special.gamma(1+k*1./a)*special.gamma(b)/\
                special.gamma(1+b+k*1./a)

    def _str(self, a, b):
        return "kum(%s,%s)" % (a,b)

    def _bnd(self, a, b):
        return 0,1

class hypgeosec(Dist):

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return .5*np.cosh(np.pi*x/2.)**-1

    def _cdf(self, x):
        return 2/np.pi*np.arctan(np.e**(np.pi*x/2.))

    def _ppf(self, q):
        return 2/np.pi*np.log(np.tan(np.pi*q/2.))

    def _mom(self, k):
        return np.abs(special.euler(k))[-1]

    def _str(self):
        return "hgs"

class logistic(Dist):

    def __init__(self, c=1):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        return np.e**-x/(1+np.e**-x)**(c+1)

    def _cdf(self, x, c):
        return (1+np.e**-x)**-c

    def _ppf(self, q, c):
        return -np.log(q**(-1/c)-1)

    def _bnd(self, c):
        return self._ppf(1e-10, c), self._ppf(1-1e-10, c)

    def _str(self, c):
        return "log(%s)" % c

class student_t(Dist):

    def __init__(self, a=1):
        Dist.__init__(self, a=a)

    def _pdf(self, x, a):
        return special.gamma(.5*a+.5)*(1+x*x/a)**(-.5*a-.5) /\
                (np.sqrt(a*np.pi)*special.gamma(.5*a))

    def _cdf(self, x, a):
        return special.stdtr(a, x)

    def _ppf(self, q, a):
        return special.stdtrit(a, q)

    def _bnd(self, a):
        return self._ppf(1e-10, a), self._ppf(1-1e-10, a)

    def _mom(self, k, a):
        if np.any(a<=k):
            raise ValueError("too high mom for student-t")
        out = special.gamma(.5*k+.5)* \
                special.gamma(.5*a-.5*k)*a**(.5*k)
        return np.where(k%2==0, out/(np.pi**.5*special.gamma(.5*a)), 0)

    def _ttr(self, k, a):
        return 0., k*a*(a-k+1.)/ ((a-2*k)*(a-2*k+2))

    def _str(self, a):
        return "stt(%s)" % a


class raised_cosine(Dist):

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return .5 + .5*np.cos(np.pi*x)

    def _cdf(self, x):
        return .5 + .5*x + np.sin(np.pi*x)/(2*np.pi)

    def _bnd(self):
        return -1,1

    def _mom(self, k):
        return np.where(k%2, 0, 2/(k+2) + 1/(k+1)*\
                special.hyp1f2((k+1)/2.), .5, (k+3)/2., -np.pi**2/4)

    def _str(self):
        return "cos"

class mvnormal(Dist):

    def __init__(self, loc=[0,0], scale=[[1,.5],[.5,1]]):
        loc, scale = np.asfarray(loc), np.asfarray(scale)
        C = np.linalg.cholesky(scale)
        Ci = np.linalg.inv(C)
        Dist.__init__(self, C=C, Ci=Ci, loc=loc,
                _advance=True, _length=len(C))

    def _cdf(self, x, graph):
        Ci, loc = graph.keys["Ci"], graph.keys["loc"]
        return sp.special.ndtr(np.dot(Ci, (x.T-loc.T).T))

    def _ppf(self, q, graph):
        return (np.dot(graph.keys["C"], sp.special.ndtri(q)).T+graph.keys["loc"].T).T

    def _pdf(self, x, graph):

        loc, C, Ci = graph.keys["loc"], graph.keys["C"], graph.keys["Ci"]
        det = np.linalg.det(np.dot(C,C.T))

        x_ = np.dot(Ci.T, (x.T-loc.T).T)
        out = np.ones(x.shape)
        out[0] =  np.e**(-.5*np.sum(x_*x_, 0))/np.sqrt((2*np.pi)**len(Ci)*det)
        return out

    def _bnd(self, x, graph):

        C, loc = graph.keys["C"], graph.keys["loc"]
        scale = np.sqrt(np.diag(np.dot(C,C.T)))
        lo,up = np.zeros((2,)+x.shape)
        lo.T[:] = (-7.5*scale+loc)
        up.T[:] = (7.5*scale+loc)
        return lo,up

    def _mom(self, k, graph):

        C, loc = graph.keys["C"], graph.keys["loc"]
        scale = np.dot(C, C.T)

        def mom(k):

            zeros = (np.sum(k,0)%2==1)+np.any(np.array(k)<0, 0)
            if np.all(zeros, 0):
                return 0.

            dim, K = k.shape
            ra = np.arange(dim).repeat(K).reshape(dim,K)

            i = np.argmax(k!=0, 0)

            out = np.zeros(k.shape[1:])
            out[:] = np.where(np.choose(i,k),
                    (np.choose(i,k)-1)*scale[i,i]*mom(k-2*(ra==i)), 1)
            for x in range(1, dim):
                out += \
                (np.choose(i,k)!=0)*(x>i)*k[x]*scale[i,x]*mom(k-(ra==i)-(ra==x))

            return out

        dim = len(loc)
        K = np.mgrid[[slice(0,_+1,1) for _ in np.max(k, 1)]]
        K = K.reshape(dim, K.size/dim)
        M = mom(K)

        out = np.zeros(k.shape[1])
        for i in range(len(M)):
            coef = np.prod(sp.misc.comb(k.T, K[:,i]).T, 0)
            diff = k.T - K[:,i]
            pos = diff>=0
            diff = diff*pos
            pos = np.all(pos, 1)
            loc_ = np.prod(loc**diff, 1)
            out += pos*coef*loc_*M[i]

        return out

    def _dep(self, graph):
        n = normal()
        out = [set([n]) for _ in range(len(self))]
        return out

    def _str(self, C, loc, **prm):
        return "mvnor(%s,%s)" % (loc, C)

class mvlognormal(Dist):

    def __init__(self, loc=[0,0], scale=[[1,.5],[.5,1]]):

        loc, scale = np.asfarray(loc), np.asfarray(scale)
        assert len(loc)==len(scale)

        dist = chaospy.dist.joint.Iid(normal(), len(loc))
        C = np.linalg.cholesky(scale)
        Ci = np.linalg.inv(C)
        Dist.__init__(self, dist=dist, loc=loc, C=C, Ci=Ci,
                scale=scale, _length=len(scale), _advance=True)

    def _cdf(self, x, graph):

        y = np.log(np.abs(x) + 1.*(x<=0))
        out = graph(np.dot(graph.keys["Ci"], (y.T-graph.keys["loc"].T).T),
                graph.dists["dist"])
        return np.where(x<=0, 0., out)

    def _ppf(self, q, graph):
        return np.e**(np.dot(graph.keys["C"], \
                graph(q, graph.dists["dist"])).T+graph.keys["loc"].T).T

    def _mom(self, k, graph):
        scale, loc = graph.keys["scale"], graph.keys["loc"]
        return np.e**(np.dot(k.T, loc).T+ \
            .5*np.diag(np.dot(k.T, np.dot(scale, k))))

    def _bnd(self, x, graph):
        loc, scale = graph.keys["loc"], graph.keys["scale"]
        up = (7.1*np.sqrt(np.diag(scale))*x.T**0 + loc.T).T
        return 0*up, np.e**up

    def _val(self, graph):
        if "dist" in graph.keys:
            return (np.dot(graph.keys["dist"].T, graph.keys["C"].T)+graph.keys["loc"].T).T
        return self

    def _dep(self, graph):

        dist = graph.dists["dist"]
        S = graph(dist)
        out = [set([]) for _ in range(len(self))]
        C = graph.keys["C"]

        for i in range(len(self)):
            for j in range(len(self)):
                if C[i,j]:
                    out[i].update(S[j])
        return out

    def _str(self, loc, C, **prm):
        print("mvlognor(%s,%s)" % (loc, C))

class mvstudentt(Dist):

    def __init__(self, a=1, loc=[0,0], scale=[[1,.5],[.5,1]]):
        loc, scale = np.asfarray(loc), np.asfarray(scale)
        C = np.linalg.cholesky(scale)
        Ci = np.linalg.inv(C)
        Dist.__init__(self, a=a, C=C, Ci=Ci, loc=loc, _length=len(C))

    def _cdf(self, x, a, C, Ci, loc):
        x = np.dot(Ci, (x.T-loc.T).T)
        return special.stdtr(a, x)

    def _ppf(self, q, a, C, Ci, loc):
        z = sp.special.stdtrit(a, q)
        out = (np.dot(C, z).T + loc.T).T
        return out

    def _pdf(self, x, a, C, Ci, loc):

        det = np.linalg.det(np.dot(C,C.T))
        k = len(C)

        x_ = np.dot(Ci.T, (x.T-loc.T).T)
        out = np.ones(x.shape)
        out[0] = special.gamma(.5*(a+k))/(special.gamma(.5*a)* \
                a**(.5*k)*np.pi**(.5*k)*det**.5*\
                (1+np.sum(x_*x_,0)/a))
        return out

    def _bnd(self, a, C, Ci, loc):

        scale = np.sqrt(np.diag(np.dot(C,C.T)))
        lo,up = np.zeros((2,len(self)))
        lo.T[:] = (-10**5*scale+loc)
        up.T[:] = (10**5*scale+loc)
        return lo,up

#      def _mom(self, k, graph):
#
#          C, loc = graph.keys["C"], graph.keys["loc"]
#          scale = np.dot(C, C.T)
#
#          def mom(k):
#
#              zeros = (np.sum(k,0)%2==1)+np.any(np.array(k)<0, 0)
#              if np.all(zeros, 0):
#                  return 0.
#
#              dim, K = k.shape
#              ra = np.arange(dim).repeat(K).reshape(dim,K)
#
#              i = np.argmax(k!=0, 0)
#
#              out = np.zeros(k.shape[1:])
#              out[:] = np.where(np.choose(i,k),
#                      (np.choose(i,k)-1)*scale[i,i]*mom(k-2*(ra==i)), 1)
#              for x in range(1, dim):
#                  out += \
#                  (np.choose(i,k)!=0)*(x>i)*k[x]*scale[i,x]*mom(k-(ra==i)-(ra==x))
#
#              return out
#
#          dim = len(loc)
#          K = np.mgrid[[slice(0,_+1,1) for _ in np.max(k, 1)]]
#          K = K.reshape(dim, K.size/dim)
#          M = mom(K)
#
#          out = np.zeros(k.shape[1])
#          for i in range(len(M)):
#              coef = np.prod(sp.misc.comb(k.T, K[:,i]).T, 0)
#              diff = k.T - K[:,i]
#              pos = diff>=0
#              diff = diff*pos
#              pos = np.all(pos, 1)
#              loc_ = np.prod(loc**diff, 1)
#              out += pos*coef*loc_*M[i]
#
#          return out

    def _dep(self, graph):
        n = student_t()
        out = [set([n]) for _ in range(len(self))]
        return out

    def _str(self, a, loc, C, **prm):
        return "mvstt(%s,%s,%s)" % (a,loc,C)

#  class Dirichlet(be.Dist):
#      """
#  Dirichlet \sim Dir(alpha)
#
#  Parameters
#  ----------
#  alpha : array_like
#      Shape parameters.
#      len(alpha)>1
#      np.all(alpha>0)
#
#  Examples
#  --------
#  >>> cp.seed(1000)
#  >>> f = cp.Dirichlet([1,2,3])
#  >>> q = [[.3,.3,.7,.7],[.3,.7,.3,.7]]
#  >>> print(f.inv(q))
#  [[ 0.06885008  0.06885008  0.21399691  0.21399691]
#   [ 0.25363028  0.47340104  0.21409462  0.39960771]]
#  >>> print(f.fwd(f.inv(q)))
#  [[ 0.3  0.3  0.7  0.7]
#   [ 0.3  0.7  0.3  0.7]]
#  >>> print(f.sample(4))
#  [[ 0.12507651  0.00904026  0.06508353  0.07888277]
#   [ 0.29474152  0.26985323  0.69375006  0.30848838]]
#  >>> print(f.mom((1,1)))
#  0.047619047619
#      """
#
#      def __init__(self, alpha=[1,1,1]):
#
#          dists = [co.beta() for _ in range(len(alpha)-1)]
#          ba.Dist.__init__(self, _dists=dists, alpha=alpha, _name="D")
#
#      def _upd(self, alpha, **prm):
#
#          alpha = alpha.flatten()
#          dim = len(alpha)-1
#          out = [None]*dim
#          _dists = prm.pop("_" + self.name)
#          cum = _dists[0]
#
#          _dists[0].upd(a=alpha[0], b=np.sum(alpha[1:], 0))
#          out[0] = _dists[0]
#          for i in range(1, dim):
#              _dists[i].upd(a=alpha[i], b=np.sum(alpha[i+1:], 0))
#              out[i] = _dists[i]*(1-cum)
#              cum = cum+out[i]
#
#          prm = dict(alpha=alpha)
#          prm["_" + self.name] = out
#          return prm
#
#      def _mom(self, k, alpha, **prm):
#
#          out = np.empty(k.shape[1:])
#          out[:] = sp.special.gamma(np.sum(alpha, 0))
#          out /= sp.special.gamma(np.sum(alpha, 0)+np.sum(k, 0))
#          out *= np.prod(sp.special.gamma(alpha[:-1]+k.T).T, 0)
#          out /= np.prod(sp.special.gamma(alpha[:-1]), 0)
#          return out
#


##NEW

class alpha(Dist):

    def __init__(self, a=1):
        Dist.__init__(self, a=a)

    def _cdf(self, x, a):
        return special.ndtr(a-1./x) / special.ndtr(a)

    def _ppf(self, q, a):
        return 1.0/(a-special.ndtri(q*special.ndtr(a)))

    def _pdf(self, x, a):
        return 1.0/(x**2)/special.ndtr(a)*np.e**(.5*(a-1.0/x)**2)/np.sqrt(2*np.pi)

    def _bnd(self, a):
        return 0,self._ppf(1-1e-10, a)

class anglit(Dist):

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return np.cos(2*x)
    def _cdf(self, x):
        return np.sin(x+np.pi/4)**2.0
    def _ppf(self, q):
        return (np.arcsin(np.sqrt(q))-np.pi/4)
    def _bnd(self):
        return -np.pi/4, np.pi/4


class bradford(Dist):

    def __init__(self, c=1):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        return  c / (c*x + 1.0) / np.log(1.0+c)
    def _cdf(self, x, c):
        return np.log(1.0+c*x) / np.log(c+1.0)
    def _ppf(self, q, c):
        return ((1.0+c)**q-1)/c
    def _bnd(self, c):
        return 0, 1

class burr(Dist):

    def __init__(self, c=1., d=1.):
        Dist.__init__(self, c=c, d=d)
    def _pdf(self, x, c, d):
        return c*d*(x**(-c-1.0))*((1+x**(-c*1.0))**(-d-1.0))
    def _cdf(self, x, c, d):
        return (1+x**(-c*1.0))**(-d**1.0)
    def _ppf(self, q, c, d):
        return (q**(-1.0/d)-1)**(-1.0/c)
    def _bnd(self, c, d):
        return 0, self._ppf(1-1e-10)

class fisk(Dist):

    def __init__(self, c=1.):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        return c*(x**(-c-1.0))*((1+x**(-c*1.0))**(-1.0))
    def _cdf(self, x, c):
        return (1+x**(-c*1.0))**(-1.0)
    def _ppf(self, q, c):
        return (q**(-1.0)-1)**(-1.0/c)
    def _bnd(self, c):
        return 0, self._ppf(1-1e-10, c)

class cauchy(Dist):

    def __init__(self):
        Dist.__init__(self)
    def _pdf(self, x):
        return 1.0/np.pi/(1.0+x*x)
    def _cdf(self, x):
        return 0.5 + 1.0/np.pi*np.arctan(x)
    def _ppf(self, q):
        return np.tan(np.pi*q-np.pi/2.0)
    def _bnd(self):
        return self._ppf(1e-10), self._ppf(1-1e-10)

class chi(Dist):

    def __init__(self, df=1):
        Dist.__init__(self, df=df)
    def _pdf(self, x, df):
        return x**(df-1.)*np.exp(-x*x*0.5)/(2.0)**(df*0.5-1)\
                /special.gamma(df*0.5)
    def _cdf(self, x, df):
        return special.gammainc(df*0.5,0.5*x*x)
    def _ppf(self, q, df):
        return np.sqrt(2*special.gammaincinv(df*0.5,q))
    def _bnd(self, df):
        return 0, self._ppf(1-1e-10, df)
    def _mom(self, k, df):
        return 2**(.5*k)*special.gamma(.5*(df+k))\
                /special.gamma(.5*df)

class dbl_gamma(Dist):

    def __init__(self, a):
        Dist.__init__(self, a=a)

    def _pdf(self, x, a):
        ax = abs(x)
        return 1.0/(2*special.gamma(a))*ax**(a-1.0) * np.exp(-ax)

    def _cdf(self, x, a):
        fac = 0.5*special.gammainc(a,abs(x))
        return np.where(x>0,0.5+fac,0.5-fac)

    def _ppf(self, q, a):
        fac = special.gammainccinv(a,1-abs(2*q-1))
        return np.where(q>0.5, fac, -fac)

    def _bnd(self, a):
        return self._ppf(1e-10, a), self._ppf(1-1e-10, a)

class dbl_weibull(Dist):

    def __init__(self, c):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        ax = np.abs(x)
        Px = c/2.0*ax**(c-1.0)*np.exp(-ax**c)
        return Px
    def _cdf(self, x, c):
        Cx1 = 0.5*np.exp(-abs(x)**c)
        return np.where(x > 0, 1-Cx1, Cx1)
    def _ppf(self, q, c):
        q_ = np.where(q>.5, 1-q, q)
        Cq1 = (-np.log(2*q_))**(1./c)
        return np.where(q>.5, Cq1, -Cq1)
    def _bnd(self, c):
        return self._ppf(1e-10, c), self._ppf(1-1e-10, c)


class erlang(Dist):


    def __init__(self, a=1):
        Dist.__init__(self, a=a)
    def _pdf(self, x, a):
        Px = (x)**(a-1.0)*np.exp(-x)/special.gamma(a)
        return Px
    def _cdf(self, x, a):
        return special.gdtr(1.0,a,x)
    def _ppf(self, q, a):
        return special.gdtrix(1.0, a, q)
    def _bnd(self, a):
        return 0, self._ppf(1-1e-10, a)

class exponweibull(Dist):

    def __init__(self, a=1, c=1):
        Dist.__init__(self, a=a, c=c)
    def _pdf(self, x, a, c):
        exc = np.exp(-x**c)
        return a*c*(1-exc)**(a-1) * exc * x**(c-1)
    def _cdf(self, x, a, c):
        exm1c = -np.expm1(-x**c)
        return (exm1c)**a
    def _ppf(self, q, a, c):
        return (-np.log1p(-q**(1.0/a)))**(1.0/c)
    def _bnd(self, a, c):
        return 0, self._ppf(1-1e-10, a, c)

class exponpow(Dist):

    def __init__(self, b=1):
        Dist.__init__(self, b=b)
    def _pdf(self, x, b):
        xbm1 = x**(b-1.0)
        xb = xbm1 * x
        return np.exp(1)*b*xbm1 * np.exp(xb - np.exp(xb))
    def _cdf(self, x, b):
        xb = x**b
        return -np.expm1(-np.expm1(xb))
    def _ppf(self, q, b):
        return pow(np.log1p(-np.log1p(-q)), 1.0/b)
    def _bnd(self, b):
        return 0,self._ppf(1-1e-10, b)

class fatiguelife(Dist):

    def __init__(self, c=0):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        return (x+1)/(2*c*np.sqrt(2*np.pi*x**3))*np.exp(-(x-1)**2/(2.0*x*c**2))
    def _cdf(self, x, c):
        return special.ndtr(1.0/c*(np.sqrt(x)-1.0/np.sqrt(x)))
    def _ppf(self, q, c):
        tmp = c*special.ndtri(q)
        return 0.25*(tmp + np.sqrt(tmp**2 + 4))**2
    def _bnd(self, c):
        return 0, self._ppf(1-1e-10, c)

class foldcauchy(Dist):

    def __init__(self, c=0):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        return 1.0/np.pi*(1.0/(1+(x-c)**2) + 1.0/(1+(x+c)**2))
    def _cdf(self, x, c):
        return 1.0/np.pi*(np.arctan(x-c) + np.arctan(x+c))
    def _bnd(self, c):
        return 0, 10**10


class foldnorm(Dist):

    def __init__(self, c=1):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        return np.sqrt(2.0/np.pi)*np.cosh(c*x)*np.exp(-(x*x+c*c)/2.0)
    def _cdf(self, x, c):
        return special.ndtr(x-c) + special.ndtr(x+c) - 1.0
    def _bnd(self, c):
        return 0, 7.5+c

class frechet(Dist):
    def __init__(self, c=1):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        return c*pow(x,c-1)*np.exp(-pow(x,c))
    def _cdf(self, x, c):
        return -np.expm1(-pow(x,c))
    def _ppf(self, q, c):
        return pow(-np.log1p(-q),1.0/c)
    def _mom(self, k, c):
        return special.gamma(1-k*1./c)
    def _bnd(self, c):
        return 0, self._ppf(1-1e-10, c)


class genexpon(Dist):
    def __init__(self, a=1, b=1, c=1):
        Dist.__init__(self, a=a, b=b, c=c)
    def _pdf(self, x, a, b, c):
        return (a+b*(-np.expm1(-c*x)))*np.exp((-a-b)*x+b*(-np.expm1(-c*x))/c)
    def _cdf(self, x, a, b, c):
        return -np.expm1((-a-b)*x + b*(-np.expm1(-c*x))/c)
    def _bnd(self, a, b, c):
        return 0, 10**10

class genextreme(Dist):
    def __init__(self, c=1):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        cx = c*x
        logex2 = np.where((c==0)*(x==x),0.0,np.log1p(-cx))
        logpex2 = np.where((c==0)*(x==x),-x,logex2/c)
        pex2 = np.exp(logpex2)
        logpdf = np.where((cx==1) | (cx==-np.inf),-np.inf,-pex2+logpex2-logex2)
        np.putmask(logpdf,(c==1) & (x==1),0.0)
        return np.exp(logpdf)

    def _cdf(self, x, c):
        loglogcdf = np.where((c==0)*(x==x),-x,np.log1p(-c*x)/c)
        return np.exp(-np.exp(loglogcdf))

    def _ppf(self, q, c):
        x = -np.log(-np.log(q))
        return np.where((c==0)*(x==x),x,-np.expm1(-c*x)/c)
    def _bnd(self, c):
        return self._ppf(1e-10, c), self._ppf(1-1e-10, c)


class gengamma(Dist):

    def __init__(self, x, a, c):
        Dist.__init__(self, a=a, c=c)
    def _pdf(self, x, a, c):
        return abs(c)* np.exp((c*a-1)*np.log(x)-x**c- special.gammaln(a))
    def _cdf(self, x, a, c):
        val = special.gammainc(a,x**c)
        cond = c + 0*val
        return np.where(cond>0,val,1-val)
    def _ppf(self, q, a, c):
        val1 = special.gammaincinv(a,q)
        val2 = special.gammaincinv(a,1.0-q)
        ic = 1.0/c
        cond = c+0*val1
        return np.where(cond > 0,val1**ic,val2**ic)
    def _mom(self, k, a, c):
        return special.gamma((c+k)*1./a)/special.gamma(c*1./a)
    def _bnd(self, a, c):
        return 0.0, self._ppf(1-1e-10, a, c)


class genhalflogistic(Dist):

    def __init__(self, c=1):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        limit = 1.0/c
        tmp = (1-c*x)
        tmp0 = tmp**(limit-1)
        tmp2 = tmp0*tmp
        return 2*tmp0 / (1+tmp2)**2
    def _cdf(self, x, c):
        limit = 1.0/c
        tmp = (1-c*x)
        tmp2 = tmp**(limit)
        return (1.0-tmp2) / (1+tmp2)
    def _ppf(self, q, c):
        return 1.0/c*(1-((1.0-q)/(1.0+q))**c)
    def _bnd(self, c):
        return 0.0, 1/np.where(c<10**-10, 10**-10, c)


class gompertz(Dist):

    def __init__(self, c):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        ex = np.exp(x)
        return c*ex*np.exp(-c*(ex-1))
    def _cdf(self, x, c):
        return 1.0-np.exp(-c*(np.exp(x)-1))
    def _ppf(self, q, c):
        return np.log(1-1.0/c*np.log(1-q))
    def _bnd(self, c):
        return 0.0, self._ppf(1-1e-10, c)


class gumbel(Dist):

    def __init__(self):
        Dist.__init__(self)
    def _pdf(self, x):
        ex = np.exp(-x)
        return ex*np.exp(-ex)
    def _cdf(self, x):
        return np.exp(-np.exp(-x))
    def _ppf(self, q):
        return -np.log(-np.log(q))
    def _bnd(self):
        return self._ppf(1e-10), self._ppf(1-1e-10)


class levy(Dist):

    def __init__(self):
        Dist.__init__(self)
    def _pdf(self, x):
        return 1/np.sqrt(2*np.pi*x)/x*np.exp(-1/(2*x))
    def _cdf(self, x):
        return 2*(1-normal._cdf(1/np.sqrt(x)))
    def _ppf(self, q):
        val = normal._ppf(1-q/2.0)
        return 1.0/(val*val)
    def _bnd(self):
        return 0.0, self._ppf(1-1e-10)


class loggamma(Dist):

    def __init__(self, c):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        return np.exp(c*x-np.exp(x)-special.gammaln(c))
    def _cdf(self, x, c):
        return special.gammainc(c, np.exp(x))
    def _ppf(self, q, c):
        return np.log(special.gammaincinv(c,q))
    def _bnd(self, c):
        return self._ppf(1e-10, c), self._ppf(1-1e-10, c)


class loglaplace(Dist):

    def __init__(self, c):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        cd2 = c/2.0
        c = np.where(x < 1, c, -c)
        return cd2*x**(c-1)
    def _cdf(self, x, c):
        return np.where(x < 1, 0.5*x**c, 1-0.5*x**(-c))
    def _ppf(self, q, c):
        return np.where(q < 0.5, (2.0*q)**(1.0/c), (2*(1.0-q))**(-1.0/c))
    def _bnd(self, c):
        return 0.0, self._ppf(1-1e-10, c)



class mielke(Dist):

    def __init__(self, k, s):
        Dist.__init__(self, k=k, s=s)
    def _pdf(self, x, k, s):
        return k*x**(k-1.0) / (1.0+x**s)**(1.0+k*1.0/s)
    def _cdf(self, x, k, s):
        return x**k / (1.0+x**s)**(k*1.0/s)
    def _ppf(self, q, k, s):
        qsk = pow(q,s*1.0/k)
        return pow(qsk/(1.0-qsk),1.0/s)
    def _bnd(self, k, s):
        return 0.0, self._ppf(1-1e-10, k, s)


class nakagami(Dist):
    def __init__(self, nu):
        Dist.__init__(self, nu=nu)
    def _pdf(self, x, nu):
        return 2*nu**nu/special.gamma(nu)*(x**(2*nu-1.0))*np.exp(-nu*x*x)
    def _cdf(self, x, nu):
        return special.gammainc(nu,nu*x*x)
    def _ppf(self, q, nu):
        return np.sqrt(1.0/nu*special.gammaincinv(nu,q))
    def _bnd(self, nu):
        return 0.0, self._ppf(1-1e-10)


class chisquared(Dist):

    def __init__(self, df, nc):
        Dist.__init__(self, df=df, nc=nc)
    def _pdf(self, x, df, nc):
        a = df/2.0
        fac = (-nc-x)/2.0 + (a-1)*np.log(x)-a*np.log(2)-special.gammaln(a)
        fac += np.nan_to_num(np.log(special.hyp0f1(a, nc * x/4.0)))
        return np.np.exp(fac)
    def _cdf(self, x, df, nc):
        return special.chndtr(x,df,nc)
    def _ppf(self, q, df, nc):
        return special.chndtrix(q,df,nc)
    def _bnd(self, df, nc):
        return 0.0, self._ppf(1-1e-10, nc)

#  class f(Dist):
#
#      def __init__(self, n=1, m=1):
#          Dist.__init__(self, n=n, m=m)
#      def _pdf(self, x, n, m):
#          lPx = m/2*np.np.log(m) + n/2*np.log(n) + (n/2-1)*np.log(x)
#          lPx -= ((n+m)/2)*np.log(m+n*x) + special.betaln(n/2,m/2)
#          return np.exp(lPx)
#      def _cdf(self, x, n, m):
#          return special.fdtr(n, m, x)
#      def _ppf(self, q, n, m):
#          return special.fdtri(n, m, q)
#      def _mom(self, k, n, m):
#          ga = special.gamma
#          return (n*1./m)**k*ga(.5*n+k)*ga(.5*m-k)/ga(.5*n)/ga(.5*m)
#      def _bnd(self, n, m):
#          return 0, self._ppf(1-1e-10, n, m)

class f(Dist):
    """A non-central F distribution continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `ncf` is::

    ncf.pdf(x, df1, df2, nc) = np.exp(nc/2 + nc*df1*x/(2*(df1*x+df2)))
                    * df1**(df1/2) * df2**(df2/2) * x**(df1/2-1)
                    * (df2+df1*x)**(-(df1+df2)/2)
                    * gamma(df1/2)*gamma(1+df2/2)
                    * L^{v1/2-1}^{v2/2}(-nc*v1*x/(2*(v1*x+v2)))
                    / (B(v1/2, v2/2) * gamma((v1+v2)/2))

    for ``df1, df2, nc > 0``.

    %(example)s

    """
    def __init__(self, dfn, dfd, nc):
        Dist.__init__(self, dfn=dfn, dfd=dfd, nc=nc)
    def _cdf(self, x, dfn, dfd, nc):
        return special.ncfdtr(dfn,dfd,nc,x)
    def _ppf(self, q, dfn, dfd, nc):
        return special.ncfdtri(dfn, dfd, nc, q)
    def _bnd(self, dfn, dfd, nc):
        return 0.0, self._ppf(1-1e-10, dfn, dfd, nc)


class nct(Dist):

    def __init__(self, df, nc):
        Dist.__init__(self, df=df, nc=nc)
    def _pdf(self, x, df, nc):
        n = df*1.0
        nc = nc*1.0
        x2 = x*x
        ncx2 = nc*nc*x2
        fac1 = n + x2
        trm1 = n/2.*np.log(n) + special.gammaln(n+1)
        trm1 -= n*np.log(2)+nc*nc/2.+(n/2.)*np.log(fac1)+special.gammaln(n/2.)
        Px = np.exp(trm1)
        valF = ncx2 / (2*fac1)
        trm1 = np.sqrt(2)*nc*x*special.hyp1f1(n/2+1,1.5,valF)
        trm1 /= (fac1*special.gamma((n+1)/2))
        trm2 = special.hyp1f1((n+1)/2,0.5,valF)
        trm2 /= (np.sqrt(fac1)*special.gamma(n/2+1))
        Px *= trm1+trm2
        return Px
    def _cdf(self, x, df, nc):
        return special.nctdtr(df, nc, x)
    def _ppf(self, q, df, nc):
        return special.nctdtrit(df, nc, q)
    def _bnd(self, df, nc):
        return self._ppf(1e-10, df, nc), self._ppf(1-1e-10, df, nc)


#  class pareto(Dist):
#      def __init__(self, c=1):
#          Dist.__init__(self, c=c)
#      def _pdf(self, x, c):
#          Px = pow(1+c*x,-1.-1./c)
#          return Px
#      def _cdf(self, x, c):
#          return 1.0 - pow(1+c*x,-1.0/c)
#      def _ppf(self, q, c):
#          vals = 1.0/c * (pow(1-q, -c)-1)
#          return vals
#      def _bnd(self, c):
#          return 1, self._ppf(1-1e-10, c)

class pareto1(Dist):
    def __init__(self, b):
        Dist.__init__(self, b=b)
    def _pdf(self, x, b):
        return b * x**(-b-1)
    def _cdf(self, x, b):
        return 1 -  x**(-b)
    def _ppf(self, q, b):
        return pow(1-q, -1.0/b)
    def _bnd(self, b):
        return 1.0, self._ppf(1-1e-10, b)
class pareto2(Dist):

    def _pdf(self, x, c):
        return c*1.0/(1.0+x)**(c+1.0)
    def _cdf(self, x, c):
        return 1.0-1.0/(1.0+x)**c
    def _ppf(self, q, c):
        return pow(1.0-q,-1.0/c)-1
    def _bnd(self, c):
        return 0.0, self._ppf(1-1e-10, c)


class powerlognorm(normal):

    def __init__(self, c, s):
        Dist.__init__(self, c=c, s=s)
    def _pdf(self, x, c, s):
        return c/(x*s)*normal._pdf(self, \
                np.log(x)/s)*pow(normal._cdf(self, -np.log(x)/s),c*1.0-1.0)

    def _cdf(self, x, c, s):
        return 1.0 - pow(normal._cdf(self, -np.log(x)/s),c*1.0)
    def _ppf(self, q, c, s):
        return np.exp(-s*normal._ppf(self, pow(1.0-q,1.0/c)))
    def _bnd(self, c, s):
        return 0.0, self._ppf(1-1e-10, c, s)


class powernorm(Dist):

    def __init__(self, c):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        return c*normal._pdf(x)* \
               (normal._cdf(-x)**(c-1.0))
    def _cdf(self, x, c):
        return 1.0-normal._cdf(-x)**(c*1.0)
    def _ppf(self, q, c):
        return -normal._ppf(pow(1.0-q,1.0/c))
    def _bnd(self, c):
        return self._ppf(1e-10, c), self._ppf(1-1e-10, c)

class wald(Dist):

    def __init__(self, mu):
        Dist.__init__(self, mu=mu)
    def _pdf(self, x, mu):
        return 1.0/np.sqrt(2*np.pi*x)*np.exp(-(1-mu*x)**2.0 / (2*x*mu**2.0))
    def _cdf(self, x, mu):
        trm1 = 1.0/mu - x
        trm2 = 1.0/mu + x
        isqx = 1.0/np.sqrt(x)
        return 1.0-normal._cdf(self, isqx*trm1)-\
                np.exp(2.0/mu)*normal._cdf(self, -isqx*trm2)
    def _bnd(self, mu):
        return 0.0, 10**10

class reciprocal(Dist):

    def __init__(self, lo=0, up=1):
        Dist.__init__(self, lo=lo, up=up)
    def _pdf(self, x, lo, up):
        return 1./(x*np.log(up/lo))
    def _cdf(self, x, lo, up):
        return np.log(x/lo)/np.log(up/lo)
    def _ppf(self, q, lo, up):
        return np.e**(q*np.log(up/lo) + np.log(lo))
    def _bnd(self, lo, up):
        return lo, up
    def _mom(self, k, lo, up):
        return ((up*np.e**k-lo*np.e**k)/(np.log(up/lo)*(k+(k==0))))**(k!=0)

class truncexpon(Dist):

    def __init__(self, b):
        Dist.__init__(self, b=b)
    def _pdf(self, x, b):
        return np.exp(-x)/(1-np.exp(-b))
    def _cdf(self, x, b):
        return (1.0-np.exp(-x))/(1-np.exp(-b))
    def _ppf(self, q, b):
        return -np.log(1-q+q*np.exp(-b))
    def _bnd(self, b):
        return 0.0, b


class truncnorm(Dist):

    def __init__(self, a, b, mu, sigma):
        Dist.__init__(self, a=a, b=b)
        self.norm = normal()*sigma+mu
        self.fa = self.norm.fwd(a)
        self.fb = self.norm.fwd(b)
    def _pdf(self, x, a, b):
        return self.norm.pdf(x) / (self.fb-self.fa)
    def _cdf(self, x, a, b):
        return (self.norm.fwd(x) - self.fa) / (self.fb-self.fa)
    def _ppf(self, q, a, b):
        return self.norm.inv(q*(self.fb-self.fa) + self.fa)
    def _bnd(self, a, b):
        return a, b


class tukeylambda(Dist):

    def __init__(self, lam):
        Dist.__init__(self, lam=lam)
    def _pdf(self, x, lam):
        Fx = (special.tklmbda(x,lam))
        Px = Fx**(lam-1.0) + ((1-Fx))**(lam-1.0)
        Px = 1.0/(Px)
        return np.where((lam <= 0) | (abs(x) < 1.0/(lam)), Px, 0.0)

    def _cdf(self, x, lam):
        return special.tklmbda(x, lam)

    def _ppf(self, q, lam):
        q = q*1.0
        vals1 = (q**lam - (1-q)**lam)/lam
        vals2 = np.log(q/(1-q))
        return np.where((lam==0)&(q==q), vals2, vals1)

    def _bnd(self, lam):
        return self._ppf(1e-10, lam), self._ppf(1-1e-10, lam)



class wrapcauchy(Dist):

    def __init__(self, c):
        Dist.__init__(self, c=c)
    def _pdf(self, x, c):
        return (1.0-c*c)/(2*np.pi*(1+c*c-2*c*np.cos(x)))
    def _cdf(self, x, c):
        output = 0.0*x
        val = (1.0+c)/(1.0-c)
        c1 = x<np.pi
        c2 = 1-c1

        xn = np.extract(c2,x)
        if (any(xn)):
            valn = np.extract(c2, np.ones_like(x)*val)
            xn = 2*np.pi - xn
            yn = np.tan(xn/2.0)
            on = 1.0-1.0/np.pi*np.arctan(valn*yn)
            np.place(output, c2, on)

        xp = np.extract(c1,x)
        if (any(xp)):
            valp = np.extract(c1, np.ones_like(x)*val)
            yp = np.tan(xp/2.0)
            op = 1.0/np.pi*np.arctan(valp*yp)
            np.place(output, c1, op)

        return output

    def _ppf(self, q, c):
        val = (1.0-c)/(1.0+c)
        rcq = 2*np.arctan(val*np.tan(np.pi*q))
        rcmq = 2*np.pi-2*np.arctan(val*np.tan(np.pi*(1-q)))
        return np.where(q < 1.0/2, rcq, rcmq)
    def _bnd(self, c):
        return 0.0, 2*np.pi

class rice(Dist):

    def __init__(self, a):
        Dist.__init__(a=a)

    def _pdf(self, x, a):
        return x*np.exp(-.5*(x*x+a*a))*special.j0(x*a)

    def _cdf(self, x, a):
        return special.chndtr(x*x, 2, a*a)

    def _ppf(self, q, a):
        return special.chdtrix(np.sqrt(q), 2, a*a)

    def _bnd(self, a):
        return 0, special.chndtrix(np.sqrt(1-1e-10), 2, a*a)

class kdedist(Dist):
    """
A distribution that is based on a kernel density estimator (KDE).
    """
    def __init__(self, kernel, lo, up):
        self.kernel = kernel
        super(kdedist, self).__init__(lo=lo, up=up)

    def _cdf(self, x, lo, up):
        cdf_vals = np.zeros(x.shape)
        for i in range(0, len(x)):
            cdf_vals[i] = [self.kernel.integrate_box_1d(0, x_i) for x_i in x[i]]
        return cdf_vals

    def _pdf(self, x, lo, up):
        return self.kernel(x)

    def _bnd(self, lo, up):
        return (lo, up)

    def sample(self, size=(), rule="R", antithetic=None,
            verbose=False, **kws):
        """
            Overwrite sample() function, because the constructed Dist that is
            based on the KDE is only working with the random sampling that is
            given by the KDE itself.
        """

        size_ = np.prod(size, dtype=int)
        dim = len(self)
        if dim>1:
            if isinstance(size, (tuple,list,np.ndarray)):
                shape = (dim,) + tuple(size)
            else:
                shape = (dim, size)
        else:
            shape = size

        out = self.kernel.resample(size_)[0]
        try:
            out = out.reshape(shape)
        except:
            if len(self)==1:
                out = out.flatten()
            else:
                out = out.reshape(dim, out.size/dim)

        return out
