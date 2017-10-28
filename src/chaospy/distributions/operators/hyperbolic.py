"""
Collection of hyerbolic operators

Example usage
-------------
"""
import numpy
from ..baseclass import Dist

class Sinh(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Sinh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return numpy.sinh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(numpy.arcsinh(x), graph.dists["dist"])/numpy.sqrt(1+x*x)

    def _cdf(self, x, graph):
        return graph(numpy.arcsinh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return numpy.sinh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return numpy.sinh(graph(numpy.arcsinh(x), graph.dists["dist"]))


class Arcsinh(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Arcsinh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return numpy.arcsinh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(numpy.sinh(x), graph.dists["dist"])*numpy.cosh(x)

    def _cdf(self, x, graph):
        return graph(numpy.sinh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return numpy.arcsinh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return numpy.arcsinh(graph(numpy.sinh(x), graph.dists["dist"]))


class Cosh(Dist):

    def __init__(self, dist):
        assert numpy.all(dist.range()>=0)
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Cos(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return numpy.cosh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(numpy.arccosh(x), graph.dists["dist"])/numpy.sqrt(x*x-1)

    def _cdf(self, x, graph):
        return graph(numpy.arccosh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return numpy.cosh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return numpy.cosh(graph(numpy.arccosh(x), graph.dists["dist"]))


class Arccosh(Dist):

    def __init__(self, dist):
        assert numpy.all(dist.range()>=1)
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Arccosh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return numpy.arccosh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(numpy.cosh(x), graph.dists["dist"])*numpy.sinh(x)

    def _cdf(self, x, graph):
        return graph(numpy.cosh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return numpy.arccosh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return numpy.arccosh(graph(numpy.cosh(x), graph.dists["dist"]))


class Tanh(Dist):

    def __init__(self, dist):
        lo,up = dist.range()
        assert numpy.all(lo>=-1) and numpy.all(up<=1)
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Tanh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return numpy.tanh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(numpy.arctanh(x), graph.dists["dist"])/numpy.sqrt(1-x*x)

    def _cdf(self, x, graph):
        return graph(numpy.arctanh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return numpy.tanh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return numpy.tanh(graph(numpy.arctanh(x), graph.dists["dist"]))


class Arctanh(Dist):

    def __init__(self, dist):
        Dist.__init__(self, dist=dist,
                _length=len(dist), _advance=True)

    def _str(self, dist):
        return "Arctanh(%s)" % dist

    def _val(self, graph):
        if "dist" in graph.keys:
            return numpy.arctanh(graph.keys["dist"])
        return self

    def _pdf(self, x, graph):
        return graph(numpy.tanh(x), graph.dists["dist"])/numpy.sinh(x)**2

    def _cdf(self, x, graph):
        return graph(numpy.tanh(x), graph.dists["dist"])

    def _ppf(self, q, graph):
        return numpy.arctanh(graph(q, graph.dists["dist"]))

    def _bnd(self, x, graph):
        return numpy.arctanh(graph(numpy.tanh(x), graph.dists["dist"]))
