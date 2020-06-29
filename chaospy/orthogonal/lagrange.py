"""Create Lagrange polynomials."""
import numpy
from scipy.special import comb
import numpoly


def lagrange_polynomial(abscissas, graded=True, reverse=True, sort=None):
    """
    Create Lagrange polynomials.

    Args:
        abscissas (numpy.ndarray):
            Sample points where the Lagrange polynomials shall be defined.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``q0**2*q1*q2``,
            ``q0*q1**2*q2`` and ``q0*q1*q2**2``, which all have exponent sum of
            5.
        reverse (bool):
            Reverse lexicographical sorting meaning that ``q0*q1**3`` is
            considered bigger than ``q0**3*q1``, instead of the opposite.

    Example:
        >>> chaospy.lagrange_polynomial([-10, 10]).round(4)
        polynomial([-0.05*q0+0.5, 0.05*q0+0.5])
        >>> chaospy.lagrange_polynomial([-1, 0, 1]).round(4)
        polynomial([0.5*q0**2-0.5*q0, -q0**2+1.0, 0.5*q0**2+0.5*q0])
        >>> poly = chaospy.lagrange_polynomial([[1, 0, 1], [0, 1, 2]])
        >>> poly.round(4)
        polynomial([-0.5*q1+0.5*q0+0.5, -q0+1.0, 0.5*q1+0.5*q0-0.5])
        >>> poly([1, 0, 1], [0, 1, 2]).round(4)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> nodes = numpy.array([[ 0.17,  0.15,  0.17,  0.19],
        ...                      [14.94, 16.69, 16.69, 16.69]])
        >>> poly = chaospy.lagrange_polynomial(nodes)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        LinAlgError: Lagrange abscissas resulted in invertible matrix
    """
    abscissas = numpy.asfarray(abscissas)
    if len(abscissas.shape) == 1:
        abscissas = abscissas.reshape(1, abscissas.size)
    dim, size = abscissas.shape

    order = 1
    while comb(order+dim, dim) < size:
        order += 1

    indices = numpoly.glexindex(0, order+1, dimensions=dim,
                                graded=graded, reverse=reverse)[:size]
    idx, idy = numpy.mgrid[:size, :size]

    matrix = numpy.prod(abscissas.T[idx]**indices[idy], -1)
    det = numpy.linalg.det(matrix)
    if det == 0:
        raise numpy.linalg.LinAlgError(
            "Lagrange abscissas resulted in invertible matrix")

    names = numpoly.variable(dim).names
    vec = numpoly.monomial(
        0, order+1, names=names, graded=graded, reverse=reverse)[:size]

    coeffs = numpy.zeros((size, size))

    if size == 1:
        out = numpoly.monomial(
            0, 1, names=names, graded=graded, reverse=reverse)*abscissas.item()

    elif size == 2:
        coeffs = numpy.linalg.inv(matrix)
        out = numpoly.sum(vec*(coeffs.T), 1)

    else:
        for i in range(size):
            if i%2 != 0:
                k = 1
            else:
                k=0
            for j in range(size):
                if k%2 == 0:
                    coeffs[i, j] += numpy.linalg.det(matrix[1:, 1:])
                else:
                    if size%2 == 0:
                        coeffs[i, j] += -numpy.linalg.det(matrix[1:, 1:])
                    else:
                        coeffs[i, j] += numpy.linalg.det(matrix[1:, 1:])
                matrix = numpy.roll(matrix, -1, axis=0) 
                k += 1
            matrix = numpy.roll(matrix, -1, axis=1) 
        coeffs /= det
        out = numpoly.sum(vec*(coeffs.T), 1)

    return out
