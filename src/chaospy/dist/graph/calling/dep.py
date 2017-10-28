# pylint: disable=protected-access
"""Backend for distribution dependency retriever."""
import numpy


def dep_call(self, dist):
    """Dependency call backend wrapper."""
    graph = self.graph
    self.dist, dist_ = dist, self.dist

    if hasattr(dist, "_dep"):
        out = dist._dep(self)
    else:
        for val in self.dist.prm.values():
            out = numpy.zeros(len(dist), dtype=bool)
            if val in graph.nodes:
                out = numpy.ones(len(dist), dtype=bool)
                break
            elif not isinstance(val, numpy.ndarray):
                graph.add_node(val)
    graph.add_node(dist, key=out)

    assert len(out) == len(dist)
    self.dist = dist_
    return out
