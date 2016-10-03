"""
Collection of trigonometric operators
"""
from .backend import Dist
import numpy as np


class Sin(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-np.pi/2) and np.all(up<=np.pi/2)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Sin(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.sin(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.arcsin(x), G.D["dist"])/np.sqrt(1-x*x)

    def _cdf(self, x, G):
        return G(np.arcsin(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.sin(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.sin(G(np.arcsin(x), G.D["dist"]))

class Arcsin(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-1) and np.all(up<=1)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Arcsin(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.arcsin(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.sin(x), G.D["dist"])*np.cos(x)

    def _cdf(self, x, G):
        return G(np.sin(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.arcsin(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.arcsin(G(np.sin(x), G.D["dist"]))


class Cos(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=0) and np.all(up<=np.pi)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Cos(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.cos(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.arccos(x), G.D["dist"])/np.sqrt(1-x*x)

    def _cdf(self, x, G):
        return 1-G(np.arccos(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.cos(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.cos(G(np.arccos(x), G.D["dist"]))



class Arccos(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-1) and np.all(up<=1)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Arccos(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.arccos(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.cos(x), G.D["dist"])*np.sin(x)

    def _cdf(self, x, G):
        return 1-G(np.cos(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.arccos(G(-q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.arccos(G(np.cos(x), G.D["dist"]))

class Tan(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-np.pi/2) and np.all(up<=np.pi/2)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Tan(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.tan(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.arctan(x), G.D["dist"])/(1-x*x)

    def _cdf(self, x, G):
        return G(np.arctan(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.tan(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.tan(G(np.arctan(x), G.D["dist"]))

class Arctan(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Arccos(%s)" % dist

    def _val(self, G):
        if "dist" in G.K:
            return np.arctan(G.K["dist"])
        return self

    def _pdf(self, x, G):
        return G(np.tan(x), G.D["dist"])/np.cos(x)**2

    def _cdf(self, x, G):
        return G(np.cos(x), G.D["dist"])

    def _ppf(self, q, G):
        return np.arctan(G(q, G.D["dist"]))

    def _bnd(self, x, G):
        return np.arctan(G(np.tan(x), G.D["dist"]))




