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

    zero = [0]*dim
    out = numpy.zeros((dim, dim) + poly.shape)

    mean = E(poly, dist)
    V_total = Var(poly, dist)
    E_cond_i = [None]*dim
    V_E_cond_i = [None]*dim
    for i in range(dim):
        zero[i] = 1
        E_cond_i[i] = E_cond(poly, zero, dist, **kws) 
        V_E_cond_i[i] = Var(E_cond_i[i], dist, **kws)
        zero[i] = 0

    for i in range(dim):

        zero[i] = 1
        for j in range(i+1, dim):

            zero[j] = 1
            E_cond_ij = E_cond(poly, zero, dist, **kws)
            out[i, j] = ((Var(E_cond_ij, dist, **kws)-V_E_cond_i[i]-V_E_cond_i[j]) /
                         (V_total+(V_total == 0))*(V_total != 0))
            out[j, i] = out[i, j]
            zero[j] = 0

        zero[i] = 0

    return out
