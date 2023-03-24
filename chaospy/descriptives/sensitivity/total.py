"""Total Sobol sensitivity index."""
import numpy
import numpoly

from ..conditional import E_cond
from ..variance import Var


def Sens_t(poly, dist, **kws):
    """
    Variance-based decomposition
    AKA Sobol' indices

    Total effect sensitivity index

    Args:
        poly (numpoly.ndpoly):
            Polynomial to find first order Sobol indices on.
        dist (Distribution):
            The distributions of the input used in ``poly``.

    Returns:
        (numpy.ndarray) :
            First order sensitivity indices for each parameters in ``poly``,
            with shape ``(len(dist),) + poly.shape``.

    Examples:
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0, q1, 10*q0*q1-1])
        >>> dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> chaospy.Sens_t(poly, dist)
        array([[0.        , 1.        , 0.        , 0.57142857],
               [0.        , 0.        , 1.        , 0.57142857]])
    """
    dim = len(dist)
    poly = numpoly.set_dimensions(poly, dim)

    out = numpy.zeros((dim,) + poly.shape, dtype=float)
    variance = Var(poly, dist, **kws)

    valids = variance != 0
    if not numpy.all(valids):
        out[:, valids] = Sens_t(poly[valids], dist, **kws)
        return out

    out[:] = variance
    for idx, unit_vec in enumerate(numpy.eye(dim, dtype=int)):
        conditional = E_cond(poly, 1 - unit_vec, dist, **kws)
        out[idx] -= Var(conditional, dist, **kws)
        out[idx] /= variance

    return out


def TotalOrderSobol(
    expansion,
    coefficients,
):
    """
    Total Sobel indices.

    Args:
        expansion (numpoly.ndpoly):
            The polynomial expansion used as basis when creating a chaos
            expansion.
        coefficients (numpy.ndarray):
            The Fourier coefficients generated whent fitting the chaos
            expansion. Typically retrieved by passing ``retall=True`` to
            ``chaospy.fit_regression`` or ``chaospy.fit_quadrature``.

    Examples:
        >>> q0, q1 = chaospy.variable(2)
        >>> expansion = chaospy.polynomial([1, q0, q1, 10*q0*q1-1])
        >>> coeffs = [1, 2, 2, 4]
        >>> chaospy.TotalOrderSobol(expansion, coeffs)
        array([0.8, 0.8])
    """
    dic = expansion.todict()
    alphas = []
    for idx in range(len(expansion)):
        expons = numpy.array([key for key, value in dic.items() if value[idx]])
        alphas.append(tuple(expons[numpy.argmax(expons.sum(1))]))
    coefficients = numpy.asfarray(coefficients)
    variance = numpy.sum(coefficients**2, axis=0)

    sens = []
    for idx in range(len(alphas[0])):
        index = numpy.array([alpha[idx] > 0 for alpha in alphas])
        sens.append(numpy.sum(coefficients[index] ** 2, axis=0) / variance)
    return numpy.array(sens)
