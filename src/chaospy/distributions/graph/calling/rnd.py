# pylint: disable=protected-access
"""Backend for random number generator."""
import numpy


def rnd_call(self, dist):
    "Sample generator call backend wrapper"

    self.counting(dist, "rnd")
    graph = self.graph
    self.dist, dist_ = dist, self.dist

    for key, value in dist.prm.items():
        if not isinstance(value, numpy.ndarray) and\
                not "key" in graph.node[value]:
            self.rnd_call(value)

    if dist.advance:
        key, _ = self.run(dist, "val")
    else:
        rnd = numpy.random.random((len(dist), self.size))
        key = self.inv_call(rnd, dist)

    assert isinstance(key, numpy.ndarray)
    graph.add_node(dist, key=key)

    self.dist = dist_

    if dist is self.root:
        out = graph.node[dist]["key"]
        return out
