"""Main operator."""
import networkx


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
        out = self(x_data)
    else:
        out = self(x_data, self.root)

    if mode=="ttr":
        out[1]**(x_data != 0)

    self.size = size
    if mode=="val":
        graph = self.graph
    else:
        graph, self.graph = self.graph, graph
    self._call = call
    self.meta = meta

    return out, graph
