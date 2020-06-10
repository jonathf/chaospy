"""Total Sobol sensitivity index."""
import numpy

from ...poly.setdim import setdim
from ..conditional import E_cond
from ..variance import Var


def Sens_t(poly, dist, **kws):
    """
    Variance-based decomposition
    AKA Sobol' indices

    Total effect sensitivity index

    Args:
        poly (chaospy.poly.ndpoly):
            Polynomial to find first order Sobol indices on.
        dist (Dist):
            The distributions of the input used in ``poly``.

    Returns:
        (numpy.ndarray) :
            First order sensitivity indices for each parameters in ``poly``,
            with shape ``(len(dist),) + poly.shape``.

    Examples:
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, x, y, 10*x*y])
        >>> dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> indices = chaospy.Sens_t(poly, dist)
        >>> print(indices)
        [[0.         1.         0.         0.57142857]
         [0.         0.         1.         0.57142857]]
    """
    dim = len(dist)
    poly = setdim(poly, dim)

    out = numpy.zeros((dim,)+poly.shape, dtype=float)
    variance = Var(poly, dist, **kws)

    valids = variance != 0
    if not numpy.all(valids):
        out[:, valids] = Sens_t(poly[valids], dist, **kws)
        return out

    out[:] = variance
    for idx, unit_vec in enumerate(numpy.eye(dim, dtype=int)):
        conditional = E_cond(poly, 1-unit_vec, dist, **kws)
        out[idx] -= Var(conditional, dist, **kws)
        out[idx] /= variance

    return out
