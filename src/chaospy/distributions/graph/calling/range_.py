# pylint: disable=protected-access
"""Backend for distribution range."""
import numpy


def range_call(self, x_data, dist):
    "range call backend wrapper"

    self.counting(dist, "range")
    assert x_data.shape == (len(dist), self.size)
    graph = self.graph
    self.dist, dist_ = dist, self.dist

    graph.add_node(dist, key=x_data)
    out = numpy.empty((2,)+x_data.shape)

    prm = self.dists.build()
    prm.update(self.keys.build())
    for key, value in prm.items():
        if not isinstance(value, numpy.ndarray):
            value_ = self.run(value, "val")[0]
            if isinstance(value_, numpy.ndarray):
                prm[key] = value_
                graph.add_node(value, key=value_)

    if dist.advance:
        out[0,:],out[1,:] = dist._bnd(x_data, self)
    else:
        lower, upper = dist._bnd(**prm)
        lower, upper = numpy.array(lower), numpy.array(upper)
        out.T[:,:,0], out.T[:,:,1] = lower.T, upper.T
    graph.add_node(dist, val=out)

    self.dist = dist_
    return numpy.array(out)
