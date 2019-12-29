"""
The script is adapted from the routine RKPW in W.B. Gragg and W.J. Harrod,
"The numerically stable reconstruction of Jacobi matrices from spectral data",
Numer. Math. 44 (1984), 317-335.

Example usage
-------------

Basic usage::

    >>> dist = chaospy.Beta(3, 6)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     10, dist, rule="clenshaw_curtis")
    >>> lanczos(3, abscissas, weights)
    array([[0.33333333, 0.39393939, 0.42680399, 0.44727847],
           [1.        , 0.02222222, 0.03471074, 0.04207915]])
"""
import numpy


def lanczos(order, abscissas, weights):

    abscissas = abscissas.astype(float)
    weights = weights.astype(float)
    assert len(weights.shape) == 1
    assert len(abscissas.shape) == 2
    assert abscissas.shape[-1] == len(weights)

    assert len(weights) > order
    alpha = abscissas[:, :order+1].copy()
    beta = numpy.zeros((len(abscissas), order+1))
    beta[:, 0] = weights[0]

    for idx in range(1, len(weights)):

        gamma = 1
        increment_new = 0

        for idy in range(min(idx, order+1)):

            increment = increment_new
            beta_new = gamma*(beta[:, idy]+weights[idx])
            sigma = 1-gamma
            gamma = numpy.where(
                beta[:, idy] <= -weights[idx], 1,
                beta[:, idy]/(beta[:, idy]+weights[idx]))
            increment_new = (
                (1-gamma)*(alpha[:, idy]-abscissas[:, idx])-gamma*increment)
            weights[idx] = numpy.where(
                gamma >= 1, sigma*beta[:, idy], increment_new**2/(1-gamma))

            alpha[:, idy] -= increment_new-increment
            beta[:, idy] = beta_new

    return numpy.asfarray([alpha.flatten(), beta.flatten()])
