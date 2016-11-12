"""
Collection of trigonometric operators
"""
from chaospy.dist.baseclass import Dist
import numpy as np


class Sin(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-np.pi/2) and np.all(up<=np.pi/2)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Sin(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.sin(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.arcsin(x), graph.dists["dist"])/np.sqrt(1-x*x)

    def _cdf(self, x, graph):
        return graph(np.arcsin(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.sin(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.sin(graph(np.arcsin(x), graph.dists["dist"]))

class Arcsin(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-1) and np.all(up<=1)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Arcsin(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.arcsin(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.sin(x), graph.dists["dist"])*np.cos(x)

    def _cdf(self, x, graph):
        return graph(np.sin(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.arcsin(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.arcsin(graph(np.sin(x), graph.dists["dist"]))


class Cos(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=0) and np.all(up<=np.pi)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Cos(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.cos(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.arccos(x), graph.dists["dist"])/np.sqrt(1-x*x)

    def _cdf(self, x, graph):
        return 1-graph(np.arccos(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.cos(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.cos(graph(np.arccos(x), graph.dists["dist"]))



class Arccos(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-1) and np.all(up<=1)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Arccos(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.arccos(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.cos(x), graph.dists["dist"])*np.sin(x)

    def _cdf(self, x, graph):
        return 1-graph(np.cos(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.arccos(graph(-q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.arccos(graph(np.cos(x), graph.dists["dist"]))

class Tan(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert np.all(lo>=-np.pi/2) and np.all(up<=np.pi/2)
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Tan(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.tan(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.arctan(x), graph.dists["dist"])/(1-x*x)

    def _cdf(self, x, graph):
        return graph(np.arctan(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.tan(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.tan(graph(np.arctan(x), graph.dists["dist"]))

class Arctan(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist, _length=len(dist),
                _advance=True)

    def _str(self, dist):
        return "Arccos(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return np.arctan(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(np.tan(x), graph.dists["dist"])/np.cos(x)**2

    def _cdf(self, x, graph):
        return graph(np.cos(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return np.arctan(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return np.arctan(graph(np.tan(x), graph.dists["dist"]))
