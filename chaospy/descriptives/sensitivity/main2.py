import numpy

from ...poly.setdim import setdim
from ..conditional import E_cond
from ..expected import E
from ..variance import Var


def Sens_m2(poly, dist, **kws):
    """
    Variance-based decomposition/Sobol' indices.

    Second order sensitivity indices.

    Args:
        poly (chaospy.poly.ndpoly):
            Polynomial to find second order Sobol indices on.
        dist (Dist):
            The distributions of the input used in ``poly``.

    Returns:
        (numpy.ndarray):
            First order sensitivity indices for each parameters in ``poly``,
            with shape ``(len(dist), len(dist)) + poly.shape``.

    Examples:
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, x*y, x*x*y*y, x*y*y*y])
        >>> dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> indices = chaospy.Sens_m2(poly, dist)
        >>> print(indices)
        [[[0.         0.         0.         0.        ]
          [0.         0.14285714 0.28571429 0.20930233]]
        <BLANKLINE>
         [[0.         0.14285714 0.28571429 0.20930233]
          [0.         0.         0.         0.        ]]]
    """
    dim = len(dist)
    poly = setdim(poly, len(dist))

    out = numpy.zeros((dim, dim) + poly.shape)
    variance = Var(poly, dist)

    valids = variance != 0
    if not numpy.all(valids):
        out[:, :, valids] = Sens_m2(poly[valids], dist, **kws)
        return out

    conditional_v = numpy.zeros((dim,)+poly.shape)
    for idx, unit_vec in enumerate(numpy.eye(dim, dtype=int)):
        conditional_e = E_cond(poly, unit_vec, dist, **kws)
        conditional_v[idx] = Var(conditional_e, dist, **kws)

    for idx, unit_vec1 in enumerate(numpy.eye(dim, dtype=int)):
        for idy, unit_vec2 in enumerate(numpy.eye(dim, dtype=int)[idx+1:], idx+1):

            conditional_e = E_cond(poly, unit_vec1+unit_vec2, dist, **kws)
            out[idx, idy] = Var(conditional_e, dist, **kws)
            out[idx, idy] -= conditional_v[idx]
            out[idx, idy] -= conditional_v[idy]
            out[idx, idy] /= variance

    # copy upper matrix triangle down to lower triangle
    indices = numpy.tril_indices(dim, -1)
    out[indices] = numpy.swapaxes(out, 0, 1)[indices]

    return out
