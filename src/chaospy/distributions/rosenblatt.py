"""
"""

import numpy as np
from . import approx


def fwd(dist, x):
    """Forward Rosenblatt transformation."""
    dim = len(dist)
    x = np.asfarray(x)
    shape = x.shape
    size = int(x.size/dim)
    x = x.reshape(dim, size)

    bnd, graph = dist.graph.run(x, "range")
    x_ = np.where(x < bnd[0], bnd[0], x)
    x_ = np.where(x_ > bnd[1], bnd[1], x_)
    out, graph = dist.graph.run(x_, "fwd")
    out = np.where(x < bnd[0], 0, out)
    out = np.where(x > bnd[1], 1, out)

    out = out.reshape(shape)
    return out


def inv(dist, q, maxiter=100, tol=1e-5):
    """Inverse Rosenblatt transformation."""
    q = np.array(q)
    assert np.all(q>=0) and np.all(q<=1), q

    dim = len(dist)
    shape = q.shape
    size = int(q.size/dim)
    q = q.reshape(dim, size)

    try:
        out, graph = dist.graph.run(q, "inv", maxiter=maxiter, tol=tol)

    except NotImplementedError:
        out, N, q_ = approx.inv(dist, q, maxiter=maxiter, tol=tol, retall=True)

        diff = np.max(np.abs(q-q_))
        print("approx %s.inv w/%d calls and eps=%g" % (dist, N, diff))

    lo,up = dist.graph.run(out, "range")[0]
    out = np.where(out.T>up.T, up.T, out.T).T
    out = np.where(out.T<lo.T, lo.T, out.T).T
    out = out.reshape(shape)

    return out
