"""Main Sobol sensitivity index."""
import numpy
import numpoly

from ..variance import Var
from ..conditional import E_cond


def Sens_m(poly, dist, **kws):
    """
    Variance-based decomposition/Sobol' indices.

    First order sensitivity indices.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to find first order Sobol indices on.
        dist (Distribution):
            The distributions of the input used in ``poly``.

    Returns:
        (numpy.ndarray):
            First order sensitivity indices for each parameters in ``poly``,
            with shape ``(len(dist),) + poly.shape``.

    Examples:
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0, q1, 10*q0*q1-1])
        >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> chaospy.Sens_m(poly, distribution)
        array([[0.        , 1.        , 0.        , 0.42857143],
               [0.        , 0.        , 1.        , 0.42857143]])
    """
    dim = len(dist)
    poly = numpoly.set_dimensions(poly, dim)

    out = numpy.zeros((dim,) + poly.shape)
    variance = Var(poly, dist, **kws)
    valids = variance != 0

    for idx, unit_vec in enumerate(numpy.eye(dim, dtype=int)):

        conditional = E_cond(poly[valids], unit_vec, dist, **kws)
        out[idx, valids] = Var(conditional, dist, **kws)
        out[idx, valids] /= variance[valids]

    return out
