"""
Graph interactive wrapper.

Used to retrieve parameters in a distribution. Comes in three flavoers::

    Dist Keys Values

Can be interacted with as follows:

Retrieve parameter::

    container["param"]

Check of paramater is in collection::

    "param" in container

Itterate over all parameters::

    for (key,param) in container

Generate dictionary with all values::

    container.build()

Identify the number of parameters available::

    len(container)
"""
import numpy


class Container(object):
    """
    """

    def __init__(self, graph):
        "Graph module"
        self.graph = graph
    def __contains__(self, key):
        raise NotImplementedError()
    def getitem(self, key):
        raise NotImplementedError()
    def __getitem__(self, key):
        return self.getitem(key)
    def build(self):
        "build a dict with all parameters"
        out = {}
        for k,v in self.graph.dist.prm.items():
            if k in self:
                out[k] = self[k]
        return out
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
            raise KeyError()
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
