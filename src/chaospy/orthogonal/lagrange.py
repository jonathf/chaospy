"""
Lagrange polynomials are not a method for creating orthogonal polynomials.
Instead it is an interpolation method for creating an polynomial expansion that
has the property that each polynomial interpolates exactly one point in space
with the value 1 and has the value 0 for all other interpolation values.
For more details, see this `article on Lagrange polynomials`_.

.. _article on Lagrange polynomials: https://en.wikipedia.org/wiki/Lagrange_polynomial
"""
import numpy
import chaospy

def lagrange_polynomial(abscissas, sort="GR"):
    """
    Create Lagrange polynomials.

    Args:
        abscissas (numpy.ndarray):
            Sample points where the Lagrange polynomials shall be defined.

    Example:
        >>> print(chaospy.around(lagrange_polynomial([-10, 10]), 4))
        [-0.05q0+0.5, 0.05q0+0.5]
        >>> print(chaospy.around(lagrange_polynomial([-1, 0, 1]), 4))
        [0.5q0^2-0.5q0, -q0^2+1.0, 0.5q0^2+0.5q0]
        >>> poly = lagrange_polynomial([[1, 0, 1], [0, 1, 2]])
        >>> print(chaospy.around(poly, 4))
        [0.5q0-0.5q1+0.5, -q0+1.0, 0.5q0+0.5q1-0.5]
        >>> print(numpy.around(poly([1, 0, 1], [0, 1, 2]), 4))
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    """
    abscissas = numpy.asfarray(abscissas)
    if len(abscissas.shape) == 1:
        abscissas = abscissas.reshape(1, abscissas.size)
    dim, size = abscissas.shape

    order = 1
    while chaospy.bertran.terms(order, dim) <= size:
        order += 1

    indices = numpy.array(chaospy.bertran.bindex(0, order-1, dim, sort)[:size])
    idx, idy = numpy.mgrid[:size, :size]

    matrix = numpy.prod(abscissas.T[idx]**indices[idy], -1)
    det = numpy.linalg.det(matrix)
    if det == 0:
        raise numpy.linalg.LinAlgError("invertible matrix required")

    vec = chaospy.poly.basis(0, order-1, dim, sort)[:size]

    coeffs = numpy.zeros((size, size))

    if size == 1:
        out = chaospy.poly.basis(0, 0, dim, sort)*abscissas.item()

    elif size == 2:
        coeffs = numpy.linalg.inv(matrix)
        out = chaospy.poly.sum(vec*(coeffs.T), 1)

    else:
        for i in range(size):
            for j in range(size):
                coeffs[i, j] += numpy.linalg.det(matrix[1:, 1:])
                matrix = numpy.roll(matrix, -1, axis=0)
            matrix = numpy.roll(matrix, -1, axis=1)
        coeffs /= det
        out = chaospy.poly.sum(vec*(coeffs.T), 1)

    return out
