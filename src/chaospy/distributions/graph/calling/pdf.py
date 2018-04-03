# pylint: disable=protected-access
"""Probability density function."""
import numpy


def pdf_call(self, x_data, dist):
    "PDF call backend wrapper"
    self.counting(dist, "pdf")
    assert x_data.shape == (len(dist), self.size)
    self.dist, dist_ = dist, self.dist

    graph = self.graph
    graph.add_node(dist, key=x_data)
    out = numpy.empty(x_data.shape)

    prm = self.dists.build()
    prm.update(self.keys.build())
    for key, value in prm.items():
        if not isinstance(value, numpy.ndarray):
            value_ = self.run(value, "val")[0]
            if isinstance(value_, numpy.ndarray):
                prm[key] = value_
                graph.add_node(value, key=value_)

    if hasattr(dist, "_pdf"):
        if dist.advance:
            out[:] = dist._pdf(x_data, self)
        else:
            out[:] = dist._pdf(x_data, **prm)
    else:
        out = approx.pdf(dist, x_data, self, **self.meta)
    graph.add_node(dist, val=out)

    self.dist = dist_
    return numpy.array(out)
