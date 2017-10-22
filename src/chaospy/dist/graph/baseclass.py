"""
Class for organizing dependencies in runtime.

Iteration with this element is required to create advanced
distributions.

To use a let a distribution be advanced, include the following
parameters on initiation::

    _advance=True       Initiates the advanced distribution
    _length=len(dist)   Te length of a distribution. Defaults to 1

Subsetting the Dist-class using the _advance feature requires the
following backend methods.
The required ones are::

    _cdf(self, x, graph)        Cumulative distribution function
    _bnd(self, x, graph)        Upper and lower bounds at location x

The following can be provided::

    _pdf(self, x, graph)        Probability density function
    _ppf(self, q, graph)        CDF inverse
    _mom(self, k, graph)        Statistical moment generator
    _ttr(self, k, graph)        TTR coefficients generator
    _dep(self, graph)           Dependency callback.
    _val(self, graph)           Value callback function. Used to
                            itterative check if a distribution has
                            been evaluated before, indirectly.
    _str(self, **prm)       Preaty print of distribution

Access parameter for distributions, input locations (or
constants), and function evaluations respectively::

    graph.dists, graph.keys, graph.values

See :mod:`dist.graph.containers` for usage.

Iterate through the graph by evaluating the distribution it is a wrapper
for::

    graph(z, dist)

where z is locations, and dist is the distribution it wraps.

Iterate through distribution's dependencies::

    for dist in graph

or get all distributions as a list::

    graph.dependencies()

print the chaospy state of graph::

    print(graph)

Make a copy of chaospy state::

    graph.copy()

Initiate an itteration process::

    graph.run(...)

Switch from PDF to forward seemlessly::

    graph.fwd_as_pdf(x, dist)

From dist.baseclass we have that there are two new methods. They are
defined as follows::

    _val(self, graph)   If enough information exists to say that the
                    distribution is evaluated given the state of
                    parameters, return that value.  Else return self.

    _dep(self, graph)   Return a list l with len(l)==len(self) with all
                    distributions it depends upon as a set in each
                    element.  A common element between two sets implies
                    dependencies.  Use graph(dist) (instead of graph(x, dist)) to
                    generate a distributions dependencies.
"""
import numpy
import networkx

from ..approx import pdf, ppf, mom
from .containers import Dists, Keys, Values


class Graph:
    """Backend engine for the distributions."""

    def __init__(self, dist):
        """
        Args:
            dist (Dist) : The root of the dependency tree.
        """
        graph = networkx.DiGraph()
        dist_collection = [dist]
        graph.add_node(dist)

        while dist_collection:

            d = dist_collection.pop()
            for key,val in d.prm.items():

                if not isinstance(val, numpy.ndarray):
                    if not (val in graph.node):
                        graph.add_node(val)
                        dist_collection.append(val)
                    graph.add_edge(d, val)

        assert networkx.is_directed_acyclic_graph(graph)

        self.graph_source = graph

        self.valmode = False
        self.size = None
        self.meta = {}
        self.graph = graph
        self.dist_collection = networkx.topological_sort(graph)
        self.root = self.dist = dist

        self.dists = Dists(self)
        self.keys = Keys(self)
        self.values = Values(self)

        self._call = None


    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __str__(self):
        graph = self.graph
        out = ""
        for dist in self.dist_collection:
            out += str(dist) + "\n"
            for key in ["key", "val"]:
                if key in graph.node[dist]:
                    out += key + "\n"
                    out += str(graph.node[dist][key]) + "\n"
            for key in ["inv", "fwd", "pdf", "rnd", "range"]:
                if key in graph.node[dist]:
                    out += key + ": "
                    out += str(graph.node[dist][key]) + "\n"
            out += "\n"

        out = out[:-1]
        return out

    def __iter__(self):
        return self.dist_collection.__iter__()

    def copy(self):
        """
        Shallow copy of graph. Distribution stays the same.
        """

        graph = Graph(self.root)
        for node in self.graph.nodes():
            graph.graph.add_node(node, **self.graph.node[node])
        graph.graph.add_edges_from(self.graph.edges())

        graph.size = self.size
        graph.meta = self.meta
        graph._call = self._call

        return graph

    def run(self, x, mode, **meta):
        """Run through network to perform an operator."""
        from .calling import call
        return call(self, x, mode, **meta)

    def counting(self, dist, mode):
        "counter function. Used for meta analysis."
        if dist in self.graph.node and \
                mode in self.graph.node[dist]:
                    self.graph.node[dist][mode] += 1
        else:
            self.graph.add_node(dist, **{mode:1})

    def pdf_call(self, x, dist):
        "PDF call backend wrapper"

        self.counting(dist, "pdf")
        assert x.shape==(len(dist), self.size)
        self.dist, dist_ = dist, self.dist

        graph = self.graph
        graph.add_node(dist, key=x)
        out = numpy.empty(x.shape)

        prm = self.dists.build()
        prm.update(self.keys.build())
        for k,v in prm.items():
            if not isinstance(v, numpy.ndarray):
                v_ = self.run(v, "val")[0]
                if isinstance(v_, numpy.ndarray):
                    prm[k] = v_
                    graph.add_node(v, key=v_)

        if hasattr(dist, "_pdf"):
            if dist.advance:
                out[:] = dist._pdf(x, self)
            else:
#                  prm = self.dists.build()
#                  prm.update(self.keys.build())
                out[:] = dist._pdf(x, **prm)
        else:
            out = pdf(dist, x, self, **self.meta)
        graph.add_node(dist, val=out)

        self.dist = dist_
        return numpy.array(out)

    def fwd_as_pdf(self, x, dist):
        """During a PDF-call, a switch to CDF might be necesarry.
        This functions initiates this switch."""

        graph_source = self.graph_source
        graph = networkx.DiGraph()
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
        out = numpy.empty(q.shape)

        prm = self.dists.build()
        prm.update(self.keys.build())
        for k,v in prm.items():
            if not isinstance(v, numpy.ndarray):
                v_ = self.run(v, "val")[0]
                if isinstance(v_, numpy.ndarray):
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
        return numpy.array(out)

    def range_call(self, x, dist):
        "range call backend wrapper"

        self.counting(dist, "range")
        assert x.shape==(len(dist), self.size)
        graph = self.graph
        self.dist, dist_ = dist, self.dist

        graph.add_node(dist, key=x)
        out = numpy.empty((2,)+x.shape)

        prm = self.dists.build()
        prm.update(self.keys.build())
        for k,v in prm.items():
            if not isinstance(v, numpy.ndarray):
                v_ = self.run(v, "val")[0]
                if isinstance(v_, numpy.ndarray):
                    prm[k] = v_
                    graph.add_node(v, key=v_)

        if dist.advance:
            _ = dist._bnd(x, self)
            out[0,:],out[1,:] = _
        else:
            lo, up = dist._bnd(**prm)
            lo, up = numpy.array(lo), numpy.array(up)
            out.T[:,:,0],out.T[:,:,1] = lo.T, up.T
        graph.add_node(dist, val=out)

        self.dist = dist_
        return numpy.array(out)

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
                out = numpy.empty((2,)+k.shape)
                prm = self.dists.build()
                prm.update(self.keys.build())
                out[0],out[1] = dist._ttr(k, **prm)
        else:
            raise NotImplementedError()

        graph.add_node(dist, val=out)

        self.dist = dist_
        return numpy.array(out)

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
                out = numpy.empty(k.shape[1:])
                prm = self.dists.build()
                prm.update(self.keys.build())
                out[:] = dist._mom(k, **prm)
        else:
            out = mom(dist, k, **self.meta)
        graph.add_node(dist, val=out)

        self.dist = dist_
        return numpy.array(out)

    def rnd_call(self, dist):
        "Sample generator call backend wrapper"

        self.counting(dist, "rnd")
        graph = self.graph
        self.dist, dist_ = dist, self.dist

        for k,v in dist.prm.items():
            if not isinstance(v, numpy.ndarray) and\
                    not "key" in graph.node[v]:
                self.rnd_call(v)

        if dist.advance:
            key, _ = self.run(dist, "val")
        else:
            rnd = numpy.random.random((len(dist),self.size))
            key = self.inv_call(rnd, dist)

        assert isinstance(key, numpy.ndarray)
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
                out = numpy.zeros(len(dist), dtype=bool)
                if val in graph.nodes:
                    out = numpy.ones(len(dist), dtype=bool)
                    break
                elif not isinstance(val, numpy.ndarray):
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
            if not isinstance(v, numpy.ndarray) and\
                    not "key" in graph.node[v]:
                        self.val_call(v)

        if hasattr(dist, "_val"):
            out = dist._val(self)

            if isinstance(out, numpy.ndarray):
                graph.add_node(dist, key=out)

        else:
            out = dist

        self.dist = dist_
        return out

    def dependencies(self):
        """
Set of node dependencies
        """
        return set(self.graph.nodes())
