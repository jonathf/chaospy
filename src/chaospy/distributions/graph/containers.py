"""
Convenience properties found on distribution object during dependency traversal.

Instead of iterating through each parameter in a distribution ``dist.prm``,
checking each what each state and acting from there, it is possible to access
a subset of parameters that fit certain criteria.

For example, to access all parameters that are also distributions, the property
``dist.graph.dists``::

    >>> dists0 = chaospy.beta(a=1, b=2).graph.dists
    >>> dists1 = chaospy.beta(a=chaospy.uniform(), b=2).graph.dists
    >>> dists2 = chaospy.beta(a=chaospy.uniform(), b=chaospy.gamma(1)).graph.dists

These objects can be used to identify how many distributions are present::

    >>> print(len(dists0), len(dists1), len(dists2))
    0 1 2

To identify specific parameters that are distributions::

    >>> print("a" in dists0, "a" in dists1, "a" in dists2)
    False True True
    >>> print("b" in dists0, "b" in dists1, "b" in dists2)
    False False True

An lastly to extract said parameters::

    >>> print(dists2["b"])
    gam(1)
    >>> print(dists1["a"])
    uni

Giving values not present or illegal values, causes exceptions::

    >>> dists0["b"]
    Traceback (most recent call last):
        ...
    KeyError: "parameter 'b' is not a distribution"
    >>> dists0["c"]
    Traceback (most recent call last):
        ...
    KeyError: "parameter key 'c' unknown; ('a', 'b') available"
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
