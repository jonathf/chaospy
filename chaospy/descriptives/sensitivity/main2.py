import numpy

from ..conditional import E_cond
from ..expected import E
from ..variance import Var


def Sens_m2(poly, dist, **kws):
    """
    Variance-based decomposition/Sobol' indices.

    Second order sensitivity indices.

    Args:
        poly (Poly):
            Polynomial to find second order Sobol indices on.
        dist (Dist):
            The distributions of the input used in ``poly``.

    Returns:
        (numpy.ndarray):
            First order sensitivity indices for each parameters in ``poly``,
            with shape ``(len(dist), len(dist)) + poly.shape``.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([1, x*y, x*x*y*y, x*y*y*y])
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
    zero = [0]*dim
    out = numpy.zeros((dim, dim) + poly.shape)

    mean = E(poly, dist)
    V_total = Var(poly, dist)
    mu_conds = [E_cond(poly, mu_cond, dist, **kws)
                for mu_cond in numpy.eye(len(dist), dtype=int)]
    mu2_conds = [Var(mu_cond, dist, **kws) for mu_cond in mu_conds]

    for i in range(dim):

        zero[i] = 1
        for j in range(i+1, dim):

            zero[j] = 1
            E_cond_ij = E_cond(poly, zero, dist, **kws)
            out[i, j] = ((Var(E_cond_ij, dist, **kws)-mu2_conds[i]-mu2_conds[j])/
                         (V_total+(V_total == 0))*(V_total != 0))
            out[j, i] = out[i, j]
            zero[j] = 0

        zero[i] = 0

    return out
