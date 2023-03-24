import numpy
import numpoly

from ..conditional import E_cond
from ..expected import E
from ..variance import Var


def Sens_m2(poly, dist, **kws):
    """
    Variance-based decomposition/Sobol' indices.

    Second order sensitivity indices.

    Args:
        poly (numpoly.ndpoly):
            Polynomial to find second order Sobol indices on.
        dist (Distribution):
            The distributions of the input used in ``poly``.

    Returns:
        (numpy.ndarray):
            First order sensitivity indices for each parameters in ``poly``,
            with shape ``(len(dist), len(dist)) + poly.shape``.

    Examples:
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0*q1, q0**3*q1, q0*q1**3])
        >>> dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> chaospy.Sens_m2(poly, dist).round(4)
        array([[[0.    , 0.    , 0.    , 0.    ],
                [0.    , 0.1429, 0.2093, 0.2093]],
        <BLANKLINE>
               [[0.    , 0.1429, 0.2093, 0.2093],
                [0.    , 0.    , 0.    , 0.    ]]])

    """
    dim = len(dist)
    poly = numpoly.set_dimensions(poly, len(dist))

    out = numpy.zeros((dim, dim) + poly.shape)
    variance = Var(poly, dist)

    valids = variance != 0
    if not numpy.all(valids):
        out[:, :, valids] = Sens_m2(poly[valids], dist, **kws)
        return out

    conditional_v = numpy.zeros((dim,) + poly.shape)
    for idx, unit_vec in enumerate(numpy.eye(dim, dtype=int)):
        conditional_e = E_cond(poly, unit_vec, dist, **kws)
        conditional_v[idx] = Var(conditional_e, dist, **kws)

    for idx, unit_vec1 in enumerate(numpy.eye(dim, dtype=int)):
        for idy, unit_vec2 in enumerate(numpy.eye(dim, dtype=int)[idx + 1 :], idx + 1):

            conditional_e = E_cond(poly, unit_vec1 + unit_vec2, dist, **kws)
            out[idx, idy] = Var(conditional_e, dist, **kws)
            out[idx, idy] -= conditional_v[idx]
            out[idx, idy] -= conditional_v[idy]
            out[idx, idy] /= variance

    # copy upper matrix triangle down to lower triangle
    indices = numpy.tril_indices(dim, -1)
    out[indices] = numpy.swapaxes(out, 0, 1)[indices]

    return out


def SecondOrderSobol(
    expansion,
    coefficients,
):
    """
    Second order variance-based decomposition/Sobol' indices.

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
        >>> chaospy.SecondOrderSobol(expansion, coeffs)
        array([[0.  , 0.16],
               [0.16, 0.  ]])
    """
    dic = expansion.todict()
    alphas = []
    for idx in range(len(expansion)):
        expons = numpy.array([key for key, value in dic.items() if value[idx]])
        alphas.append(tuple(expons[numpy.argmax(expons.sum(1))]))
    coefficients = numpy.asfarray(coefficients)
    variance = numpy.sum(coefficients**2, axis=0)

    sens = numpy.zeros(
        (len(alphas[0]), len(alphas[0])) + coefficients.shape[1:], dtype=float
    )
    for idx in range(len(alphas[0])):
        for idy in range(idx):
            index = numpy.array(
                [
                    bool(
                        alpha[idx]
                        and alpha[idx]
                        and not any(alpha[:idy] + alpha[idy:idx] + alpha[idx + 1 :])
                    )
                    for alpha in alphas
                ]
            )
            sens[idx, idy] = sens[idy, idx] = (
                numpy.sum(coefficients[index] ** 2, axis=0) / variance
            )
    return sens
