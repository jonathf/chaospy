"""Convert three terms recurrence coefficients into quadrature rules."""
import numpy
import scipy.linalg


def coefficients_to_quadrature(coeffs):
    """
    Construct Gaussian quadrature abscissas and weights from three terms
    recurrence coefficients.

    Examples:
        >>> distribution = chaospy.Normal(0, 1)
        >>> coeffs = chaospy.construct_recurrence_coefficients(4, distribution)
        >>> print(numpy.around(coeffs, 4))
        [[[0. 0. 0. 0. 0.]
          [0. 1. 2. 3. 4.]]]
        >>> abscissas, weights = chaospy.coefficients_to_quadrature(coeffs)
        >>> print(numpy.around(abscissas, 4))
        [[-2.857  -1.3556 -0.      1.3556  2.857 ]]
        >>> print(numpy.around(weights, 4))
        [[0.0113 0.2221 0.5333 0.2221 0.0113]]
    """
    coeffs = numpy.asfarray(coeffs)
    if len(coeffs.shape) == 2:
        coeffs = coeffs.reshape(1, 2, -1)
    assert len(coeffs.shape) == 3, "shape %s not allowed" % coeffs.shape
    assert coeffs.shape[-1] >= 1
    abscissas = []
    weights = []
    for coeff in coeffs:

        if numpy.any(coeff[1] < 0) or numpy.any(numpy.isnan(coeff)):
            raise numpy.linalg.LinAlgError(
                "Invalid recurrence coefficients can not be used for "
                "constructing Gaussian quadrature rule")

        order = len(coeff[0])
        if order:
            bands = numpy.zeros((2, order))
            bands[0, :] = coeff[0, :order]
            bands[1, :-1] = numpy.sqrt(coeff[1, 1:order])
            vals, vecs = scipy.linalg.eig_banded(bands, lower=True)

            abscissa, weight = vals.real, vecs[0, :]**2
            indices = numpy.argsort(abscissa)
            abscissa, weight = abscissa[indices], weight[indices]

        else:
            abscissa, weight = numpy.array([coeffs[dim][0, 0]]), numpy.array([1.])

        abscissas.append(abscissa)
        weights.append(weight)

    return abscissas, weights
