"""Main Sobol sensitivity index."""
import numpy

from ...poly.setdim import setdim
from ..variance import Var
from ..conditional import E_cond


def Sens_m(poly, dist, **kws):
    """
    Variance-based decomposition/Sobol' indices.

    First order sensitivity indices.

    Args:
        poly (chaospy.poly.ndpoly):
            Polynomial to find first order Sobol indices on.
        dist (Dist):
            The distributions of the input used in ``poly``.

    Returns:
        (numpy.ndarray):
            First order sensitivity indices for each parameters in ``poly``,
            with shape ``(len(dist),) + poly.shape``.

    Examples:
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, x, y, 10*x*y])
        >>> dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> indices = chaospy.Sens_m(poly, dist)
        >>> print(indices)
        [[0.         1.         0.         0.42857143]
         [0.         0.         1.         0.42857143]]
    """
    dim = len(dist)
    poly = setdim(poly, dim)

    zero = [0]*dim
    out = numpy.zeros((dim,) + poly.shape)
    V = Var(poly, dist, **kws)
    for i in range(dim):
        zero[i] = 1
        out[i] = Var(E_cond(poly, zero, dist, **kws),
                     dist, **kws)/(V+(V == 0))*(V != 0)
        zero[i] = 0
    return out
