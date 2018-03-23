# pylint: disable=protected-access
"""Backend for forward Rosenblatt transformations."""
import numpy


def fwd_call(self, x_data, dist):
    "forward call backend wrapper"
    self.counting(dist, "fwd")
    assert x_data.shape == (len(dist), self.size)
    self.dist, dist_ = dist, self.dist

    self.graph.add_node(dist, key=x_data)
    out = numpy.empty(x_data.shape)

    prm = self.dists.build()
    prm.update(self.keys.build())
    for key, value in prm.items():

        if not isinstance(value, numpy.ndarray):
            post_value = self.run(value, "val")[0]
            if isinstance(post_value, numpy.ndarray):
                prm[key] = post_value
                graph.add_node(value, key=post_value)

    if dist.advance:
        out[:] = dist._cdf(x_data, self)
    else:
        out[:] = dist._cdf(x_data, **prm)

    self.graph.add_node(dist, val=out)

    self.dist = dist_
    return numpy.array(out)
