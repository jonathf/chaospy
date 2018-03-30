"""
Convenience properties found on distribution object during dependency traversal.

Instead of iterating through each parameter in a distribution ``dist.prm``,
checking each what each state and acting from there, it is possible to access
a subset of parameters that fit certain criteria.
"""
import numpy


class Container(object):
    """Baseclass for quick access attributes."""

    def __init__(self, graph):
        self.graph = graph

    def __contains__(self, key):
        raise NotImplementedError()

    def getitem(self, key):
        raise NotImplementedError()

    def __getitem__(self, key):
        if key not in self.graph.dist.prm:
            raise KeyError(
                "parameter key '%s' unknown; %s available" % (
                    key, tuple(self.graph.dist.prm)))
        return self.getitem(key)

    def build(self):
        "build a dict with all parameters"
        return {key: self[key] for key in self.graph.dist.prm if key in self}

    def __iter__(self):
        return self.build().values().__iter__()

    def __str__(self):
        return str(self.build())

    def __len__(self):
        return len(self.build())


class Dists(Container):
    """Contains all distributions."""

    def __contains__(self, key):
        return not isinstance(self.graph.dist.prm[key], numpy.ndarray)

    def getitem(self, key):
        if not key in self:
            raise KeyError("parameter '%s' is not a distribution" % key)
        return self.graph.dist.prm[key]


class Keys(Container):
    """
    Contains all values.

    Either as constants or as substitutes of distributions.
    """

    def __contains__(self, key):

        out = False
        val = self.graph.dist.prm[key]
        if isinstance(val, numpy.ndarray) or \
                "key" in self.graph.graph.node[val]:
            out = True

        return out

    def getitem(self, key):

        if not key in self:
            print(key)
            print(self.graph.dist.prm)
            raise KeyError()
        val = self.graph.dist.prm[key]

        if isinstance(val, numpy.ndarray):
            return val

        gkey = self.graph.dist.prm[key]
        return self.graph.graph.node[gkey]["key"]

class Values(Container):
    """Contains all evaluations of distributions."""

    def __contains__(self, key):
        out = self.graph.dist.prm[key]
        return  not isinstance(out, numpy.ndarray) and \
                "val" in self.graph.graph.node[out]

    def getitem(self, key):
        if not key in self:
            raise KeyError()
        return self.graph.graph.node[self.graph.dist.prm[key]]["val"]
