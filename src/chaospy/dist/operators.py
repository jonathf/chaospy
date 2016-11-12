"""
Collection of standard operators. They include:
    add     Addition
    div     Division
    log     Logarithm
    mul     Multiplication
    neg     Sign inversion
    pow     Exponenent function
    trunk   Truncation (independent distributions only)

They are all advanced variables, so take a look as dist.graph for
details on their implementation.
"""
import numpy as np
from scipy.misc import comb

from .backend import Dist

__all__ = [
"log", "log10", "logn",
"trunk", "pow",
"add", "mul", "div", "neg"
        ]


class Add(Dist):

    def __init__(self, A, B):
        a = not isinstance(A, Dist) or 1 and len(A)
        b = not isinstance(B, Dist) or 1 and len(B)
        length = max(a, b)

        Dist.__init__(self, A=A, B=B,
                _length=length, _advance=True)

    def _str(self, A, B):
        return "(" + str(A) + "+" + str(B) + ")"

    def _val(self, G):
        if len(G.K)==2:
            return G.K["A"] + G.K["B"]
        return self

    def _bnd(self, x, G):
        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]
        lo,up = G((x.T-num.T).T, dist)
        return (lo.T+num.T).T, (up.T+num.T).T

    def _cdf(self, x, G):
        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]
        return G((x.T-num.T).T, dist)

    def _pdf(self, x, G):
        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]
        return G((x.T-num.T).T, dist)

    def _ppf(self, q, G):
        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]
        return (G(q, dist).T + num.T).T


    def _mom(self, k, G):
        if len(G.D)==2 and G.D["A"].dependent(G.D["B"]):
            raise NotImplementedError("dependency")

        dim = len(self)
        kmax = np.max(k, 1)+1
        K = np.mgrid[[slice(0,_,1) for _ in kmax]]
        K = K.reshape(dim, K.size/dim)

        A = []
        if "A" in G.D:  A.append(G(K, G.D["A"]))
        else:           A.append(G.K["A"]**K.T)
        if "B" in G.D:  A.append(G(K, G.D["B"]))
        else:           A.append(G.K["B"]**K.T)

        if dim==1:
            A[0] = A[0].flatten()
            A[1] = A[1].flatten()
            k = k.flatten()

        out = 0.
        for i in range(K.shape[1]):
            I = K.T[i]
            coef = comb(k.T, I)
            a0 = A[0][i]
            a1 = A[1][k-i]
            out = out + coef*a0*a1*(I<=k.T)

        if dim>1:
            out = np.prod(out, 1)
        return out

    def _ttr(self, n, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]

        a,b = G(n, dist)
        return a+num, b

def add(A, B):

    if A is B: return mul(2, A)
    if isinstance(A, Dist):

        if not isinstance(B, Dist):
            B = np.array(B)
            if B.size==1 and B==0:
                return A

    elif isinstance(B, Dist):
        A = np.array(A)
        if A.size==1 and A==0:
            return B

    else:
        return A+B

    return Add(A=A, B=B)


class Mul(Dist):

    def __init__(self, A, B):
        a = not isinstance(A, Dist) or 1 and len(A)
        b = not isinstance(B, Dist) or 1 and len(B)
        length = max(a,b)
        Dist.__init__(self, A=A, B=B,
                _length=length, _advance=True)


    def _bnd(self, x, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]

        num = (num + 1.*(num==0))
        x = (x.T/num.T).T
        lo_,up_ = G(x, dist)
        lo = (num.T*(lo_.T*(num.T>0) + up_.T*(num.T<0))).T
        up = (num.T*(up_.T*(num.T>0) + lo_.T*(num.T<0))).T

        return lo,up


    def _cdf(self, x, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]
        x = (x.T/(num.T + 1.*(num.T==0))).T
        out = G(x, dist)
        out = (out.T*(num.T>0) + (1.-out.T)*(num.T<0) + \
                1.*(x.T>num.T)*(num.T==0)).T
        return out

    def _ppf(self, q, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]
        q = (q*(num>0) + (1.-q)*(num<=0))
        return G(q, dist)*num

    def _pdf(self, x, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]
        num = np.where(num, num, np.inf)
        return np.abs(G(x*1./num, dist)/num)

    def _mom(self, k, G):

        if len(G.D)==2 and G.D["A"].dependent(G.D["B"]):
            raise NotImplementedError("dependency")

        A = []
        if "A" in G.D:  A.append(G(k, G.D["A"]))
        else:           A.append((G.K["A"].T**k.T).T)
        if "B" in G.D:  A.append(G(k, G.D["B"]))
        else:           A.append((G.K["B"].T**k.T).T)

        return np.prod(A[0]*A[1], 0)


    def _ttr(self, n, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]

        a,b = G(n, dist)
        return a*num, b*num*num


    def _val(self, G):

        if len(G.K)==2:
            return G.K["A"]*G.K["B"]
        return self

    def _str(self, A, B):
        return str(A) + "*" + str(B)


class Div(Dist):

    def __init__(self, A, B):
        a = not isinstance(A, Dist) or 1 and len(A)
        b = not isinstance(B, Dist) or 1 and len(B)
        if isinstance(A, Dist):
            assert not np.any(np.prod(A.range(), 0)<0)
        if isinstance(B, Dist):
            assert not np.any(np.prod(B.range(), 0)<0)
        length = max(a,b)
        Dist.__init__(self, A=A, B=B,
                _length=length, _advance=True)

    def _pdf(self, x, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
            num = (num.T*np.ones(x.shape[::-1])).T

            p0 = G.fwd_as_pdf(x*0, dist)
            x, num = np.where(x, x, 1), \
                    np.where(x, num, np.inf*np.sign(num))
            p1 = G(num/x, dist)
            out = np.abs((1-2*p0)*p1*num/x**2)
        else:
            num, dist = G.K["B"], G.D["A"]
            num = (num.T*np.ones(x.T.shape)).T
            num = np.where(num, num, np.inf)
            out = np.abs(G(x*1./num, dist)/num)
        return out

    def _cdf(self, x, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
            num = (num.T*np.ones(x.shape[::-1])).T
            p0 = G.copy()(x*0, dist)
            x, num = np.where(x, x, 1), \
                    np.where(x, num, np.inf*np.sign(num))
            p1 = G(num/x, dist)
            out = (1-p0)*(1-p1) + p0*p1
        else:
            num, dist = G.K["B"], G.D["A"]
            x = (x.T/(num.T + 1.*(num.T==0))).T
            out = G(x, dist)
            out = (out.T*(num.T>0) + (1.-out.T)*(num.T<0) + \
                    1.*(x.T>num.T)*(num.T==0)).T
        return out

    def _ppf(self, q, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]
            num = (num.T*np.ones(q.shape[::-1])).T
            q = (q*(num>0) + (1.-q)*(num<=0))
        return G(q, dist)*num

    def _bnd(self, x, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]

        num = (num + 1.*(num==0))
        x = (x.T/num.T).T
        lo_,up_ = G(x, dist)
        lo = (num.T*(lo_.T*(num.T>0) + up_.T*(num.T<0))).T
        up = (num.T*(up_.T*(num.T>0) + lo_.T*(num.T<0))).T

        return lo,up

    def _mom(self, k, G):

        if len(G.D)==2 and G.D["A"].dependent(G.D["B"]):
            raise NotImplementedError("dependency")

        A = []
        if "A" in G.D:  A.append(G(k, G.D["A"]))
        else:           A.append((G.K["A"].T**k.T).T)
        if "B" in G.D:  A.append(G(k, G.D["B"]))
        else:           A.append((G.K["B"].T**k.T).T)

        return np.prod(A[0]*A[1], 0)


    def _ttr(self, n, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
        else:
            num, dist = G.K["B"], G.D["A"]

        a,b = G(n, dist)
        return a*num, b*num*num


    def _val(self, G):

        if len(G.K)==2:
            return G.K["A"]*G.K["B"]
        return self

    def _str(self, A, B):
        return str(A) + "*" + str(B)

def div(A, B):
    if A is B:
        return 1.

    if isinstance(A, Dist):
        if not isinstance(B, Dist):
            return mul(A, 1./B)

    elif not isinstance(B, Dist):
        return A/B

    return Div(A, B)


class Mvmul(Dist):

    def __init__(self, dist, C):
        Dist.__init__(self, dist=dist, C=C,
                Ci=np.linalg.inv(C),
                _length=len(dist), _advance=True)

    def _cdf(self, x, G):
        return G(np.dot(G.K["Ci"], x), G.D["dist"])

    def _ppf(self, q, G):
        return np.dot(G.K["C"], G(q, G.D["dist"]))

    def _bnd(self, x, G):


        bnd = G(x, G.D["dist"])
        C = G.K["C"]
        lo = (np.dot(C, bnd[0]).T).T
        up = (np.dot(C, bnd[1]).T).T

        wrong = lo>up
        out = np.where(wrong, up, lo), np.where(wrong, lo, up)
        return out

    def _val(self, G):

        if "dist" in G.K:
            return np.dot(G.K["dist"].T, G.K["C"].T).T
        return self

    def _str(self, C, Ci, dist):
        return str(dist) + "*" + str(C)

    def _dep(self, G):
        dist = G.D["dist"]
        S = G(dist)

        out = [set([]) for _ in range(len(self))]
        C = G.K["C"]

        for i in range(len(self)):
            for j in range(len(self)):
                if C[i,j]:
                    out[i].update(S[j])

        return out

def mul(A, B):

    if A is B:
        return pow(A, 2)

    if isinstance(A, Dist):

        if not isinstance(B, Dist):
            B = np.array(B)
            if B.size==1:
                if B==1:
                    return A
                if B==0:
                    return 0.

    elif isinstance(B, Dist):

        A = np.array(A)
        if A.size==1:
            if A==1:
                return B
            if A==0:
                return 0.

    else:
        return A*B

    a = not isinstance(A, Dist) or 1 and len(A)
    b = not isinstance(B, Dist) or 1 and len(B)
    length = max(a,b)
    if length==1:
        return Mul(A, B)
    return Mvmul(dist=A, C=B)



class Neg(Dist):

    def __init__(self, A):
        Dist.__init__(self, A=A,
                _length=len(A), _advance=True)

    def _bnd(self, x, G):
        return -G(-x, G.D["A"])[::-1]

    def _pdf(self, x, G):
        return G(-x, G.D["A"])

    def _cdf(self, x, G):
        return 1-G(-x, G.D["A"])

    def _ppf(self, q, G):
        return -G(1-q, G.D["A"])

    def _mom(self, k, G):
        return (-1)**np.sum(k, 0)*G(k, G.D["A"])

    def _ttr(self, k, G):
        a,b = G(k, G.D["A"])
        return -a,b

    def _str(self, A):
        return "(-" + str(A) + ")"

    def _val(self, G):
        if "A" in G.K:
            return -G.K["A"]
        return self

def neg(A):
    if not isinstance(A, Dist):
        return -A
    return Neg(A)



class Log(Dist):

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        assert np.all(dist.range()>=0)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Log(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.log(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.e**x, G.D["dist"])*np.e**x

    def _cdf(self, x, G):
        return G(np.e**x, G.D["dist"])

    def _ppf(self, q, G):
        return np.log(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        lo,up = G(np.e**x, G.D["dist"])
        return np.log(lo), np.log(up)

def log(dist):
    return Log(dist)

class Log10(Dist):

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        assert np.all(dist.range()>=0)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Log10(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.log10(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(10**x, G.D["dist"])*np.log(10)*10**x

    def _cdf(self, x, G):
        return G(10**x, G.D["dist"])

    def _ppf(self, q, G):
        return np.log10(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        lo,up = G(10**x, G.D["dist"])
        return np.log10(lo), np.log10(up)

def log10(dist):
    return Log10(dist)


class Logn(Dist):

    def __init__(self, dist, n=2):
        assert isinstance(dist, Dist)
        assert np.all(dist.range()>=0)
        assert n>0 and n!=1
        Dist.__init__(self, dist=dist, n=n,
                _length=len(dist), _advance=True)

    def _str(self, dist, n):
        return "Logn(%s,&s)" % (dist, n)

    def _val(self, G):
        if "dist" in G.K:
            return np.log(G.K["dist"])/np.log(G.K["n"])
        return self

    def _pdf(self, x, G):
        n = G.K["n"]
        return G(n**x, G.D["dist"])*np.log(n)*n**x

    def _cdf(self, x, G):
        return G(G.K["n"]**x, G.D["dist"])

    def _ppf(self, q, G):
        return np.log(G(q, G.D["dist"]))/G.K["n"]

    def _bnd(self, x, G):
        n = G.K["n"]
        lo,up = G(n**x, G.D["dist"])
        return np.log(lo)/np.log(n),\
                np.log(up)/np.log(n)

def logn(dist, n):
    return Logn(dist, n)

class Trunk(Dist):

    def __init__(self, A, B):

        if isinstance(A, Dist):
            assert not A.dependent()
            assert np.all(A.range()[0] < B)
        if isinstance(B, Dist):
            assert not B.dependent()
            assert np.all(A < B.range()[1])

        a = not isinstance(A, Dist) or 1 and len(A)
        b = not isinstance(B, Dist) or 1 and len(B)
        length = max(a, b)

        Dist.__init__(self, A=A, B=B,
                _length=length, _advance=True)

    def _str(self, A, B):
        return "(%s<%s)" % (A, B)

    def _pdf(self, x, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
            num = (num.T*np.ones(x.shape[::-1])).T
            norm = dist.fwd(num)
            out = (G(x, dist)-norm)/(1-norm)
        else:
            num, dist = G.K["B"], G.D["A"]
            num = (num.T*np.ones(x.shape[::-1])).T
            out = G(x, dist)/dist.fwd(num)

        return out

    def _cdf(self, x, G):

        G_ = G.copy()
        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
            num = (num.T*np.ones(x.shape[::-1])).T
            u1 = G(x, dist)
            u2 = G_(num, dist)
            out = (u1-u2)/(1-u2)
        else:
            num, dist = G.K["B"], G.D["A"]
            num = (num.T*np.ones(x.shape[::-1])).T
            u1 = G(x, dist)
            u2 = G_(num, dist)
            out = (u1)/u2

        return out

    def _ppf(self, q, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
            num = (num.T*np.ones(q.shape[::-1])).T
            u = dist.fwd(num)
            out = G(q*(1-u)+u, dist)
        else:
            num, dist = G.K["B"], G.D["A"]
            num = (num.T*np.ones(q.shape[::-1])).T
            u = dist.fwd(num)
            out = G(q*u, dist)

        return out

    def _bnd(self, x, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
            num = (num.T*np.ones(x.shape[::-1])).T
            x = np.where(x>num, num, x)
            lo,up = G(x, dist)
            lo = np.max([num, lo], 0)
        else:
            num, dist = G.K["B"], G.D["A"]
            num = (num.T*np.ones(x.shape[::-1])).T
            x = np.where(x<num, num, x)
            lo,up = G(x, dist)
            up = np.min([num, up], 0)

        return lo, up

def trunk(A, B):

    if not isinstance(A, Dist) and not isinstance(B, Dist):
        return A<B
    return Trunk(A, B)



class Pow(Dist):

    def __init__(self, A, B):

        a = 1 if not isinstance(A, Dist) else len(A)
        b = 1 if not isinstance(B, Dist) else len(B)
        length = max(a,b)
        Dist.__init__(self, A=A, B=B,
                _length=length, _advance=True)

    def _bnd(self, x, G):

        if "A" in G.K and "B" in G.D:

            num, dist = G.K["A"], G.D["B"]

            assert np.all(num>=0), "root of negative number"

            y = np.where(x<0, -np.inf,
                    np.log(np.where(x>0, x, 1)))/\
                        np.log(num*(1.-(num==1)))

            bnd = num**G(y, dist)
            correct = bnd[0]<bnd[1]
            bnd_ = np.empty(bnd.shape)
            bnd_[0] = np.where(correct, bnd[0], bnd[1])
            bnd_[1] = np.where(correct, bnd[1], bnd[0])

        else:

            num, dist = G.K["B"], G.D["A"]
            y = np.sign(x)*np.abs(x)**(1./num)
            y[(x == 0.)*(num < 0)] = np.inf

            bnd = G(y, dist)
            assert np.all((num % 1 == 0) + (bnd[0] >= 0)), \
                    "root of negative number"

            pair = num % 2 == 0
            bnd_ = np.empty(bnd.shape)
            bnd_[0] = np.where(pair*(bnd[0]*bnd[1]<0), 0, bnd[0])
            bnd_[0] = np.where(pair*(bnd[0]*bnd[1]>0), \
                    np.min(np.abs(bnd), 0), bnd_[0])**num
            bnd_[1] = np.where(pair, np.max(np.abs(bnd), 0),
                    bnd[1])**num

            bnd_[0], bnd_[1] = np.where(
                bnd_[0] < bnd_[1], bnd_[0], bnd_[1]
            ), np.where(
                bnd_[0] < bnd_[1], bnd_[1], bnd_[0]
            )

        return bnd_


    def _cdf(self, x, G):

        if "A" in G.K and "B" in G.D:

            num, dist = G.K["A"], G.D["B"]
            assert np.all(num>0), "imaginary result"

            y = np.log(np.abs(x) + 1.*(x<=0))/\
                    np.log(np.abs(num)+1.*(num == 1))

            out = G(y, dist)
            out = np.where(x<=0, 0., out)

        else:

            num, dist = G.K["B"], G.D["A"]
            y = np.sign(x)*np.abs(x)**(1./num)
            pairs = np.sign(x**num) != -1

            _1 = G.copy()(-y, dist)
            out = G(y, dist)
            out = np.where(num < 0, 1-out, out - pairs*_1)

        return out

    def _ppf(self, q, G):

        if "A" in G.K and "B" in G.D:
            num, dist = G.K["A"], G.D["B"]
            out = num**G(q, dist)
        else:
            num, dist = G.K["B"], G.D["A"]
            out = G(q, dist)**num
        return out

    def _pdf(self, x, G):

        if "A" in G.K and "B" in G.D:

            num, dist = G.K["A"], G.D["B"]
            assert np.all(num>0), "imaginary result"
            x_ = np.where(x<=0, -np.inf,
                    np.log(x + 1.*(x<=0))/np.log(num+1.*(num==1)))
            num_ = np.log(num+1.*(num==1))*x
            num_ = num_ + 1.*(num_==0)

            out = G(x_, dist)/num_

        else:

            num, dist = G.K["B"], G.D["A"]
            x_ = np.sign(x)*np.abs(x)**(1./num -1)
            x = np.sign(x)*np.abs(x)**(1./num)
            pairs = np.sign(x**num) == 1

            G_ = G.copy()
            out = G(x, dist)
            if np.any(pairs):
                out = out + pairs*G_(-x, dist)
            out = np.sign(num)*out * x_ / num
            out[np.isnan(out)] = np.inf

        return out

    def _mom(self, k, G):

        if "B" in G.K and not np.any(G.K["B"] % 1):
            out = G(k*np.array(G.K["B"], dtype=int), G.D["A"])
        else:
            raise NotImplementedError()
        return out

    def _val(self, G):
        if len(G.K)==2:
            return G.K["A"]**G.K["B"]
        return self

    def _str(self, A, B):
        return "(%s)**(%s)" % (A,B)


def pow(A, B):

    if isinstance(A, Dist):

        if not isinstance(B, Dist):
            B = np.array(B)
            if B.size==1 and B==1:
                return A

    elif not isinstance(B, Dist):
        return A+B

    return Pow(A=A, B=B)


class Composit(Dist):

    def __init__(self, dist, N):
        N = int(N)
        Dist.__init__(self, dist=dist, N=N,
                _length=len(dist)*N)

    def _cdf(self, x, dist, N):

        shape = x.shape
        LO,UP = dist.range()
        dim = len(dist)
        x = x.reshape(dim, N, x.size/dim/N)
        t = np.linspace(0,1,N+1)

        Q = dist.fwd(x)
        out = np.empty(x.shape)
        up = LO
        for n in range(N):
            lo, up = up, t[n+1]*(UP-LO)+LO
            out[:,n] = ((Q[:,n].T-lo)/(up-lo)).T

        out = out.reshape(shape)
        return out

    def _ppf(self, q, dist, N):

        dim = len(dist)
        shape = q.shape
        q = q.reshape(dim, N, q.size/dim/N)
        t = np.linspace(0,1,N+1)
        for n in range(N):
            q[:,n] = q[:,n]*(t[n+1]-t[n]) + t[n]
        return dist.inv(q).reshape(shape)


    def _bnd(self, dist, N):

        dim = len(dist)
        lo,up = dist.range()

        out = np.empty((2,dim,N))
        for d in range(dim):
            t = np.linspace(lo[d], up[d], N+1)
            out[0,d], out[1,d] = t[:-1], t[1:]
        out = out.reshape(2, dim*N)
        return out

    def _str(self, dist, N):
        return "%s/%d" % (dist, N)

#  from chaospy.quadrature import momgen
#  class Custom(Dist):
#  
#      def __init__(self, func, dist, order=10, **kws):
#          sample = np.array(func(dist.sample()))
#          length = sample.size
#          self.func = func
#          Dist.__init__(self, dist=dist, _length=length,
#                  _advance=True)
#          self._mom = momgen(func, order, self, **kws)
#  
#      def _cdf(self, x, G):
#          pass
#  
#      def _ppf(self, q, G):
#          return np.array(self.func(G(q, G.D["dist"])))
#  
#      def _bnd(self, x, G):
#          grid = np.mgrid[(slice(0,2,1),)*len(G.D["dist"])]
#          y = np.array(self.func(grid))
#          y = y.reshape(len(self), y.size/len(self))
#          return np.min(y,-1), np.max(y,-1)

