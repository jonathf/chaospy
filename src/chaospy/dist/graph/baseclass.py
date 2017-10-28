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

    graph.dependencies

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

from . import calling, main, containers


def construct_graph(dist):
    graph = networkx.DiGraph()
    graph.add_node(dist)
    dist_collection = [dist]
    while dist_collection:

        cur_dist = dist_collection.pop()

        values = (value for value in cur_dist.prm.values()
                    if not isinstance(value, numpy.ndarray))
        for value in values:
            if value not in graph.node:
                graph.add_node(value)
                dist_collection.append(value)
            graph.add_edge(cur_dist, value)

    assert networkx.is_directed_acyclic_graph(graph)
    return graph


class Graph:
    """Backend engine for the distributions."""

    def __init__(self, dist):
        """
        Args:
            dist (Dist) : The root of the dependency tree.
        """
        graph = construct_graph(dist)

        self.graph_source = graph
        self.graph = graph

        self.valmode = False
        self.size = None
        self.meta = {}
        self.root = self.dist = dist

        # to be deprecated:
        self.dists = containers.Dists(self)
        self.keys = containers.Keys(self)
        self.values = containers.Values(self)

        self._call = None

    @property
    def dists_(self):
        """All distributions found in the parameters."""
        return {key: value for key, value in self.dist.prm.items()
                if not isinstance(value, numpy.ndarray)}

    @property
    def keys_(self):
        """All values, either constants or distribution substitutes."""
        return {key: value for key, value in self.dist.prm.items()
                if isinstance(value, numpy.ndarray)
                or "key" in self.graph.graph.node[value]}

    @property
    def values_(self):
        """Contains all evaluations of distributions."""
        return {key: value for key, value in self.dist.prm.items()
                if isinstance(value, numpy.ndarray)
                or "val" in self.graph.graph.node[value]}

    @property
    def dist_collection(self):
        """Create collection of all distribution in dependency graph."""
        return networkx.topological_sort(self.graph)

    def __call__(self, *args, **kwargs):
        return self._call(self, *args, **kwargs)

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
        """Shallow copy of graph. Distribution stays the same."""
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
        return main.call(self, x, mode, **meta)

    def counting(self, dist, mode):
        """counter function. Used for meta analysis."""
        if dist in self.graph.node and \
                mode in self.graph.node[dist]:
            self.graph.node[dist][mode] += 1
        else:
            self.graph.add_node(dist, **{mode:1})

    def fwd_as_pdf(self, x, dist):
        """
        During a PDF-call, a switch to CDF might be necessary. This functions
        initiates this switch.
        """
        graph_source = self.graph_source
        graph = networkx.DiGraph()
        for node in graph_source.nodes():
            graph.add_node(node)
        graph.add_edges_from(graph_source.edges())
        graph, self.graph = self.graph, graph

        self._call = calling.fwd_call
        out = self(x, dist)

        graph, self.graph = self.graph, graph
        self._call = calling.pdf_call

        return out

    @property
    def dependencies(self):
        """Set of node dependencies."""
        return set(self.graph.nodes())
