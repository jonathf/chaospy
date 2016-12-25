"""
Collection of hyerbolic operators
"""
import numpy as np
from chaospy.dist.baseclass import Dist

class Sinh(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Sinh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.sinh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.arcsinh(x), graph.dists["dist"])/np.sqrt(1+x*x)

    def _cdf(self, x, graph):
        return graph(np.arcsinh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.sinh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.sinh(graph(np.arcsinh(x), graph.dists["dist"]))


class Arcsinh(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Arcsinh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.arcsinh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.sinh(x), graph.dists["dist"])*np.cosh(x)

    def _cdf(self, x, graph):
        return graph(np.sinh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.arcsinh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.arcsinh(graph(np.sinh(x), graph.dists["dist"]))


class Cosh(Dist):

    def __init__(self, dist):
        assert np.all(dist.range()>=0)
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Cos(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.cosh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.arccosh(x), graph.dists["dist"])/np.sqrt(x*x-1)

    def _cdf(self, x, graph):
        return graph(np.arccosh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.cosh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.cosh(graph(np.arccosh(x), graph.dists["dist"]))


class Arccosh(Dist):

    def __init__(self, dist):
        assert np.all(dist.range()>=1)
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Arccosh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.arccosh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.cosh(x), graph.dists["dist"])*np.sinh(x)

    def _cdf(self, x, graph):
        return graph(np.cosh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.arccosh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.arccosh(graph(np.cosh(x), graph.dists["dist"]))


class Tanh(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-1) and np.all(up<=1)
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Tanh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.tanh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.arctanh(x), graph.dists["dist"])/np.sqrt(1-x*x)

    def _cdf(self, x, graph):
        return graph(np.arctanh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.tanh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.tanh(graph(np.arctanh(x), graph.dists["dist"]))


class Arctanh(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Arctanh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.arctanh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.tanh(x), graph.dists["dist"])/np.sinh(x)**2

    def _cdf(self, x, graph):
        return graph(np.tanh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.arctanh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.arctanh(graph(np.tanh(x), graph.dists["dist"]))
