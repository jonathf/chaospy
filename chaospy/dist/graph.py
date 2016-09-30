r"""
Constructing an transitional operator requires to construct the
distributions different from a stand-alone variable. To illustrate how
to construct these types of variables, consider the following example.
Let :math:`Q=A+B`, where one of :math:`A` and :math:`B` is a random
variable, and the other a scalar. Which variable is what dependents on
the user setup of the variable. Assuming that :math:`A` is the random
variable, we have that

.. math::

    F_{Q\mid B}(q\mid b) = \mathbb P {q\leq Q\mid B\!=\!b} =
    \mathbb P {q\leq AB\mid B\!=\!b}

    = \mathbb P {\tfrac qb\leq A\mid B\!=\!b} =
    F_{A\mid B}(\tfrac qb\mid b).

Because of symmetry the distribution will be the same, but with
:math:`A` and :math:`B` substituted.

This is required when trying to use operators on multivariate variables. To
create such a variable with ``construct`` provide an additional ``length``
keyword argument specifying the length of a distribution.
"""

import numpy as np

import networkx as nx
from .approx import pdf, ppf, mom


class Graph():
    """
    Class for organizing dependencies in runtime.

    Iteraction with this element is required to create advanced
    distributions.

    To use a let a distribution be advanced, include the following
    parameters on initiation::

        _advance=True       Initiates the advanced distribution
        _length=len(dist)   Te length of a distribution. Defaults to 1

    Subsetting the Dist-class using the _advance feature requires the
    following backend methods.
    The required ones are::

        _cdf(self, x, G)        Cumulative distribution function
        _bnd(self, x, G)        Upper and lower bounds at location x

    The following can be provided::

        _pdf(self, x, G)        Probability density function
        _ppf(self, q, G)        CDF inverse
        _mom(self, k, G)        Statistical moment generator
        _ttr(self, k, G)        TTR coefficients generator
        _dep(self, G)           Dependency callback.
        _val(self, G)           Value callback function. Used to
                                itterative check if a distribution has
                                been evaluated before, indirectly.
        _str(self, **prm)       Preaty print of distribution

    Here G is the graph. It is used as follows.

    Access parameter for distributions, input locations (or
    constants), and function evaluations respectively::

        G.D, G.K, G.V

    See Container for respective usage usage.

    Iterate through the graph by evaluating the distribution it is a wrapper
    for::

        G(z, dist)

    where z is locations, and dist is the distribution it wraps.

    Iterate through distribution's dependencies::

        for dist in G

    or get all distributions as a list::

        G.dependencies()

    print the chaospy state of graph::

        print(G)

    Make a copy of chaospy state::

        G.copy()

    Initiate an itteration process::

        G.run(...)

    Switch from PDF to forward seemlessly::

        G.fwd_as_pdf(x, dist)

    From dist.backend we have that there are two new methods. They are
    defined as follows::

        _val(self, G)   If enough information exists to say that the
                        distribution is evaluated given the state of
                        parameters, return that value.  Else return self.

        _dep(self, G)   Return a list l with len(l)==len(self) with all
                        distributions it depends upon as a set in each
                        element.  A common element between two sets implies
                        dependencies.  Use G(dist) (instead of G(x, dist)) to
                        generate a distributions dependencies.
    """

    def __init__(self, dist):
        """
        Args:
            dist (Dist) : The root of the dependency tree.
        """

        graph = nx.DiGraph()
        L = [dist]
        graph.add_node(dist)

        while L:

            d = L.pop()
            for key,val in d.prm.items():

                if not isinstance(val, np.ndarray):
                    if not (val in graph.node):
                        graph.add_node(val)
                        L.append(val)
                    graph.add_edge(d, val)

        assert nx.is_directed_acyclic_graph(graph)

        self.graph_source = graph

        self.valmode = False
        self.size = None
        self.meta = {}
        self.graph = graph
        self.L = nx.topological_sort(graph)
        self.root = self.dist = dist

        self.D = Dists(self)
        self.K = Keys(self)
        self.V = Vals(self)

        self._call = None


    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)


    def __str__(self):
        graph = self.graph
        out = ""
        for dist in self.L:
            out += str(dist) + "\n"
            for key in ["key", "val"]:
                if key in graph.node[dist]:
                    out += key + "\n"
                    out += str(graph.node[dist][key]) + "\n"
            for key in ["inv", "fwd", "pdf", "rnd",
                    "range"]:
                if key in graph.node[dist]:
                    out += key + ": "
                    out += str(graph.node[dist][key]) + "\n"
            out += "\n"

        out = out[:-1]
        return out

    def __iter__(self):
        return self.L.__iter__()

    def copy(self):
        """
        Shallow copy of graph. Distribution stays the same.
        """

        G = Graph(self.root)
        for node in self.graph.nodes():
            G.graph.add_node(node, **self.graph.node[node])
        G.graph.add_edges_from(self.graph.edges())

        G.size = self.size
        G.meta = self.meta
        G._call = self._call

        return G


    def run(self, x, mode, **meta):
        """
        Run through network to perform an operator.

        Using this during another call is possible, but care has to be
        taken to ensure that the chaospy graph state is compatible with the
        new call.

        Available modes to run in:
        ------- -------------------------------------------
        "fwd"   Forward
        "inv"   Inverse
        "range" Lower and upper limits
        "ttr"   Three terms recursion coefficient generator
        "mom"   Raw statistical moments
        "dep"   Dependencies
        "val"   Values
        ------- -------------------------------------------

        Args:
            x (np.ndarray) : Input variable with shape=(D,K), where D is the
                    number of dimensions in dist.
            mode (str) : What type of operator to perform.
        **meta
            Keyword argument passed to approximation method if run.
        """
        if mode in ("rnd", "dep", "val"):
            self.size, size = x, self.size
        else:
            self.size, size = x.shape[-1], self.size
        self.meta, meta = meta, self.meta

        if mode!="val":
            graph_source = self.graph_source
            graph = nx.DiGraph()
            for node in graph_source.nodes():
                graph.add_node(node)
            graph.add_edges_from(graph_source.edges())
            graph, self.graph = self.graph, graph

        call = self._call
        if mode=="fwd":
            self._call = self.fwd_call
        elif mode=="pdf":
            self._call = self.pdf_call
        elif mode=="inv":
            self._call = self.inv_call
        elif mode=="range":
            self._call = self.range_call
        elif mode=="ttr":
            self._call = self.ttr_call
        elif mode=="mom":
            self._call = self.mom_call
        elif mode=="dep":
            self._call = self.dep_call
        elif mode=="rnd":
            self._call = self.rnd_call
        elif mode=="val":
            self._call = self.val_call
        else:
            raise ValueError("unknown mode")

        if mode!="val":
            self.dist = self.root

        if mode in ("rnd","dep"):
            out = self(self.root)

        elif mode=="val":
            out = self(x)
        else:
            out = self(x, self.root)

        if mode=="ttr":
            out[1]**(x!=0)

        self.size = size
        if mode=="val":
            graph = self.graph
        else:
            graph, self.graph = self.graph, graph
        self._call = call
        self.meta = meta

        return out, graph

    def counting(self, dist, mode):
        "counter function. Used for meta analysis."
        if dist in self.graph.node and \
                mode in self.graph.node[dist]:
                    self.graph.node[dist][mode] += 1
        else:
            self.graph.add_node(dist, **{mode:1})

    def fwd_call(self, x, dist):
        "forward call backend wrapper"

        self.counting(dist, "fwd")
        assert x.shape==(len(dist), self.size)
        self.dist, dist_ = dist, self.dist

        graph = self.graph
        graph.add_node(dist, key=x)
        out = np.empty(x.shape)

        prm = self.D.build()
        prm.update(self.K.build())
        for k,v in prm.items():
            if not isinstance(v, np.ndarray):
                v_ = self.run(v, "val")[0]
                if isinstance(v_, np.ndarray):
                    prm[k] = v_
                    graph.add_node(v, key=v_)

        if dist.advance:
            out[:] = dist._cdf(x, self)
        else:
#              prm = self.D.build()
#              prm.update(self.K.build())
            out[:] = dist._cdf(x, **prm)

        graph.add_node(dist, val=out)

        self.dist = dist_
        return np.array(out)

    def pdf_call(self, x, dist):
        "PDF call backend wrapper"

        self.counting(dist, "pdf")
        assert x.shape==(len(dist), self.size)
        self.dist, dist_ = dist, self.dist

        graph = self.graph
        graph.add_node(dist, key=x)
        out = np.empty(x.shape)

        prm = self.D.build()
        prm.update(self.K.build())
        for k,v in prm.items():
            if not isinstance(v, np.ndarray):
                v_ = self.run(v, "val")[0]
                if isinstance(v_, np.ndarray):
                    prm[k] = v_
                    graph.add_node(v, key=v_)

        if hasattr(dist, "_pdf"):
            if dist.advance:
                out[:] = dist._pdf(x, self)
            else:
#                  prm = self.D.build()
#                  prm.update(self.K.build())
                out[:] = dist._pdf(x, **prm)
        else:
            out = pdf(dist, x, self, **self.meta)
        graph.add_node(dist, val=out)

        self.dist = dist_
        return np.array(out)

    def fwd_as_pdf(self, x, dist):
        """During a PDF-call, a switch to CDF might be necesarry.
        This functions initiates this switch."""

        graph_source = self.graph_source
        graph = nx.DiGraph()
        for node in graph_source.nodes():
            graph.add_node(node)
        graph.add_edges_from(graph_source.edges())
        graph, self.graph = self.graph, graph

        self._call = self.fwd_call
        out = self(x, dist)

        graph, self.graph = self.graph, graph
        self._call = self.pdf_call

        return out


    def inv_call(self, q, dist):
        "inverse call backend wrapper"

        self.counting(dist, "inv")
        assert q.shape==(len(dist), self.size)
        self.dist, dist_ = dist, self.dist

        graph = self.graph
        graph.add_node(dist, val=q)
        out = np.empty(q.shape)

        prm = self.D.build()
        prm.update(self.K.build())
        for k,v in prm.items():
            if not isinstance(v, np.ndarray):
                v_ = self.run(v, "val")[0]
                if isinstance(v_, np.ndarray):
                    prm[k] = v_
                    graph.add_node(v, key=v_)

        if hasattr(dist, "_ppf"):
            if dist.advance:
                out[:] = dist._ppf(q, self)
            else:
                out[:] = dist._ppf(q, **prm)
        else:
            out,N,q_ = ppf(dist, q, self,
                    retall=1, **self.meta)
        graph.add_node(dist, key=out)

        self.dist = dist_
        return np.array(out)

    def range_call(self, x, dist):
        "range call backend wrapper"

        self.counting(dist, "range")
        assert x.shape==(len(dist), self.size)
        graph = self.graph
        self.dist, dist_ = dist, self.dist

        graph.add_node(dist, key=x)
        out = np.empty((2,)+x.shape)

        prm = self.D.build()
        prm.update(self.K.build())
        for k,v in prm.items():
            if not isinstance(v, np.ndarray):
                v_ = self.run(v, "val")[0]
                if isinstance(v_, np.ndarray):
                    prm[k] = v_
                    graph.add_node(v, key=v_)

        if dist.advance:
            _ = dist._bnd(x, self)
            out[0,:],out[1,:] = _
        else:
            lo, up = dist._bnd(**prm)
            lo, up = np.array(lo), np.array(up)
            out.T[:,:,0],out.T[:,:,1] = lo.T, up.T
        graph.add_node(dist, val=out)

        self.dist = dist_
        return np.array(out)

    def ttr_call(self, k, dist):
        "TTR call backend wrapper"

        assert k.shape==(len(dist), self.size)
        graph = self.graph
        self.dist, dist_ = dist, self.dist

        graph.add_node(dist, key=k)
        if hasattr(dist, "_ttr"):
            if dist.advance:
                out = dist._ttr(k, self)
            else:
                out = np.empty((2,)+k.shape)
                prm = self.D.build()
                prm.update(self.K.build())
                out[0],out[1] = dist._ttr(k, **prm)
        else:
            raise NotImplementedError()

        graph.add_node(dist, val=out)

        self.dist = dist_
        return np.array(out)

    def mom_call(self, k, dist):
        "Moment generator call backend wrapper"
        assert len(k)==len(dist)
        graph = self.graph
        self.dist, dist_ = dist, self.dist

        graph.add_node(dist, key=k)
        if hasattr(dist, "_mom"):
            if dist.advance:
                out = dist._mom(k, self)
            else:
                out = np.empty(k.shape[1:])
                prm = self.D.build()
                prm.update(self.K.build())
                out[:] = dist._mom(k, **prm)
        else:
            out = mom(dist, k, **self.meta)
        graph.add_node(dist, val=out)

        self.dist = dist_
        return np.array(out)

    def rnd_call(self, dist):
        "Sample generator call backend wrapper"

        self.counting(dist, "rnd")
        graph = self.graph
        self.dist, dist_ = dist, self.dist

        for k,v in dist.prm.items():
            if not isinstance(v, np.ndarray) and\
                    not "key" in graph.node[v]:
                self.rnd_call(v)

        if dist.advance:
            key, _ = self.run(dist, "val")
        else:
            rnd = np.random.random((len(dist),self.size))
            key = self.inv_call(rnd, dist)

        assert isinstance(key, np.ndarray)
        graph.add_node(dist, key=key)

        self.dist = dist_

        if dist is self.root:
            out = graph.node[dist]["key"]
            return out


    def dep_call(self, dist):
        "Dependency call backend wrapper"

        graph = self.graph
        self.dist, dist_ = dist, self.dist

        if hasattr(dist, "_dep"):
            out = dist._dep(self)
        else:
            for val in self.prm.values():
                out = np.zeros(len(dist), dtype=bool)
                if val in graph.nodes:
                    out = np.ones(len(dist), dtype=bool)
                    break
                elif not isinstance(val, np.ndarray):
                    graph.add_node(val)
        graph.add_node(dist, key=out)

        assert len(out)==len(dist)
        self.dist = dist_
        return out

    def val_call(self, dist):
        "Value callback wrapper"

        graph = self.graph

        if "key" in graph.node[dist]:
            return graph.node[dist]["key"]

        self.dist, dist_ = dist, self.dist
        for k,v in dist.prm.items():
            if not isinstance(v, np.ndarray) and\
                    not "key" in graph.node[v]:
                        self.val_call(v)

        if hasattr(dist, "_val"):
            out = dist._val(self)

            if isinstance(out, np.ndarray):
                graph.add_node(dist, key=out)

        else:
            out = dist

        self.dist = dist_
        return out

    def dependencies(self):
        """
Set of node dependencies
        """
        return set(self.G.nodes())


class Container(object):
    """
    Graph interactive wrapper.

    Used to retrieve parameters in a distribution. Comes in three flavoers:

        D   Dist
        K   Keys
        V   Vals

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

    def __init__(self, G):
        "Graph module"
        self.G = G
    def __contains__(self, key):
        raise NotImplementedError()
    def getitem(self, key):
        raise NotImplementedError()
    def __getitem__(self, key):
        return self.getitem(key)
    def build(self):
        "build a dict with all parameters"
        out = {}
        for k,v in self.G.dist.prm.items():
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
        return not isinstance(self.G.dist.prm[key], np.ndarray)

    def getitem(self, key):
        if not key in self:
            raise KeyError()
        return self.G.dist.prm[key]

class Keys(Container):
    """
    Contains all values.

    Either as constants or as substitutes of distributions.
    """

    def __contains__(self, key):

        out = False
        val = self.G.dist.prm[key]
        if isinstance(val, np.ndarray) or \
                "key" in self.G.graph.node[val]:
            out = True

#          else:
#              val_ = self.G.run(val, "val")
#              if isinstance(val_, np.ndarray):
#                  out = True
#                  self.G.graph.add_node(val, key=val_)

        return out

    def getitem(self, key):

        if not key in self:
            raise KeyError()
        val = self.G.dist.prm[key]

        if isinstance(val, np.ndarray):
            return val

        gkey = self.G.dist.prm[key]
        return self.G.graph.node[gkey]["key"]

class Vals(Container):
    """Contains all evaluations of distributions."""

    def __contains__(self, key):
        out = self.G.dist.prm[key]
        return  not isinstance(out, np.ndarray) and \
                "val" in self.G.graph.node[out]

    def getitem(self, key):
        if not key in self:
            raise KeyError()
        return self.G.graph.node[self.G.dist.prm[key]]["val"]
