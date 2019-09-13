"""Main Sobol sensitivity index."""
import numpy

from ..variance import Var
from ..conditional import E_cond


def Sens_m(poly, dist, **kws):
    """
    Variance-based decomposition/Sobol' indices.

    First order sensitivity indices.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to find first order Sobol indices on.
        dist (chaospy.Dist):
            The distributions of the input used in ``poly``.

    Returns:
        (numpy.ndarray):
            First order sensitivity indices for each parameters in ``poly``,
            with shape ``(len(dist),) + poly.shape``.

    Examples:
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([1, x, y, 10*x*y])
        >>> dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> indices = chaospy.Sens_m(poly, dist)
        >>> print(indices)
        [[0.         1.         0.         0.42857143]
         [0.         0.         1.         0.42857143]]
    """
    out = numpy.concatenate([
        Var(E_cond(poly, cond, dist, **kws), dist)[numpy.newaxis]
        for cond in numpy.eye(len(dist), dtype=int)
    ], axis=0)
    variance = Var(poly, dist, **kws)
    out *= numpy.where(variance, 1./variance, 0.)
    return out
