# pylint: disable=protected-access
"""Inverse Rosenblatt transformation function."""
import numpy



def inv_call(self, q_data, dist):
    "inverse call backend wrapper"

    self.counting(dist, "inv")
    assert q_data.shape == (len(dist), self.size)
    self.dist, dist_ = dist, self.dist

    graph = self.graph
    graph.add_node(dist, val=q_data)
    out = numpy.empty(q_data.shape)

    prm = self.dists.build()
    prm.update(self.keys.build())
    for key, value in prm.items():
        if not isinstance(value, numpy.ndarray):
            value_ = self.run(value, "val")[0]
            if isinstance(value_, numpy.ndarray):
                prm[key] = value_
                graph.add_node(value, key=value_)

    if hasattr(dist, "_ppf"):
        if dist.advance:
            out[:] = dist._ppf(q_data, self)
        else:
            out[:] = dist._ppf(q_data, **prm)
    else:
        out, _, _ = approx.ppf(
            dist, q_data, self, retall=1, **self.meta)
    graph.add_node(dist, key=out)

    self.dist = dist_
    return numpy.array(out)
