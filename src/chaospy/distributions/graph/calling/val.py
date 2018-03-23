# pylint: disable=protected-access
"""Backend for value callback function."""
import numpy


def val_call(self, dist):
    """Value callback wrapper."""
    graph = self.graph

    if "key" in graph.node[dist]:
        return graph.node[dist]["key"]

    self.dist, dist_ = dist, self.dist
    for value in dist.prm.values():
        if not isinstance(value, numpy.ndarray) and\
                not "key" in graph.node[value]:
            val_call(self, value)

    if hasattr(dist, "_val"):
        out = dist._val(self)

        if isinstance(out, numpy.ndarray):
            graph.add_node(dist, key=out)

    else:
        out = dist

    self.dist = dist_
    return out
