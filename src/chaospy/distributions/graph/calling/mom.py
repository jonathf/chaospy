# pylint: disable=protected-access
"""Backend for raw statistical moment generator."""
import numpy



def mom_call(self, expon, dist):
    "Moment generator call backend wrapper"
    assert len(expon) == len(dist)
    graph = self.graph
    self.dist, dist_ = dist, self.dist

    graph.add_node(dist, key=expon)
    if hasattr(dist, "_mom"):
        if dist.advance:
            out = dist._mom(expon, self)
        else:
            out = numpy.empty(expon.shape[1:])
            prm = self.dists.build()
            prm.update(self.keys.build())
            out[:] = dist._mom(expon, **prm)
    else:
        out = approx.mom(dist, expon, **self.meta)
    graph.add_node(dist, val=out)

    self.dist = dist_
    return numpy.array(out)
