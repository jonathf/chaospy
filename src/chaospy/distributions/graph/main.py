# pylint: disable=protected-access
"""Main operator."""
import networkx

from . import calling


def call(self, x_data, mode, **meta):
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
        x_data (numpy.ndarray) : Input variable with shape=(D,K), where D is the
                number of dimensions in dist.
        mode (str) : What type of operator to perform.
    **meta
        Keyword argument passed to approximation method if run.
    """
    if mode in ("rnd", "dep", "val"):
        self.size, size = x_data, self.size
    else:
        self.size, size = x_data.shape[-1], self.size
    self.meta, meta = meta, self.meta

    if mode != "val":
        graph_source = self.graph_source
        graph = networkx.DiGraph()
        for node in graph_source.nodes():
            graph.add_node(node)
        graph.add_edges_from(graph_source.edges())
        graph, self.graph = self.graph, graph

    stored_call = self._call
    assert mode in calling.CALL_FUNCTIONS, "unknown mode %s" % mode
    self._call = calling.CALL_FUNCTIONS[mode]

    if mode != "val":
        self.dist = self.root

    if mode in ("rnd", "dep"):
        out = self(self.root)

    elif mode == "val":
        out = self(x_data)
    else:
        out = self(x_data, self.root)

    # if mode == "ttr":
    #     out[1] = out[1]**(x_data != 0)

    self.size = size
    if mode == "val":
        graph = self.graph
    else:
        graph, self.graph = self.graph, graph
    self._call = stored_call
    self.meta = meta

    return out, graph
