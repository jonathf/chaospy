"""
Collection of hyerbolic operators
"""
import numpy as np
from .backend import Dist

class Sinh(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Sinh(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.sinh(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.arcsinh(x), G.D["dist"])/np.sqrt(1+x*x)

    def _cdf(self, x, G):
        return G(np.arcsinh(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.sinh(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.sinh(G(np.arcsinh(x), G.D["dist"]))


class Arcsinh(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Arcsinh(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.arcsinh(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.sinh(x), G.D["dist"])*np.cosh(x)

    def _cdf(self, x, G):
        return G(np.sinh(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.arcsinh(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.arcsinh(G(np.sinh(x), G.D["dist"]))


class Cosh(Dist):

    def __init__(self, dist):
        assert np.all(dist.range()>=0)
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Cos(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.cosh(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.arccosh(x), G.D["dist"])/np.sqrt(x*x-1)

    def _cdf(self, x, G):
        return G(np.arccosh(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.cosh(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.cosh(G(np.arccosh(x), G.D["dist"]))


class Arccosh(Dist):

    def __init__(self, dist):
        assert np.all(dist.range()>=1)
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Arccosh(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.arccosh(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.cosh(x), G.D["dist"])*np.sinh(x)

    def _cdf(self, x, G):
        return G(np.cosh(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.arccosh(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.arccosh(G(np.cosh(x), G.D["dist"]))


class Tanh(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-1) and np.all(up<=1)
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Tanh(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.tanh(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.arctanh(x), G.D["dist"])/np.sqrt(1-x*x)

    def _cdf(self, x, G):
        return G(np.arctanh(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.tanh(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.tanh(G(np.arctanh(x), G.D["dist"]))


class Arctanh(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Arctanh(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.arctanh(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.tanh(x), G.D["dist"])/np.sinh(x)**2

    def _cdf(self, x, G):
        return G(np.tanh(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.arctanh(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.arctanh(G(np.tanh(x), G.D["dist"]))
