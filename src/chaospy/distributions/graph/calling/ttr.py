# pylint: disable=protected-access
"""Backend for three terms recursion generator."""
import numpy


def ttr_call(self, order, dist):
    "TTR call backend wrapper"

    assert order.shape == (len(dist), self.size)
    graph = self.graph
    self.dist, dist_ = dist, self.dist

    graph.add_node(dist, key=order)
    if hasattr(dist, "_ttr"):
        if dist.advance:
            out = dist._ttr(order, self)
        else:
            out = numpy.empty((2,)+order.shape)
            prm = self.dists.build()
            prm.update(self.keys.build())
            out[0], out[1] = dist._ttr(order, **prm)
    else:
        raise NotImplementedError(
            "No `_ttr` method found for %s" % dist)

    graph.add_node(dist, val=out)

    self.dist = dist_
    return numpy.array(out)
