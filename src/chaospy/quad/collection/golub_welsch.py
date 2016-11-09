"""
Implementation of the Golub-Welsh algorithm.
"""
import numpy
import scipy.linalg

import chaospy.quad


def quad_golub_welsch(order, dist, accuracy=100, **kws):
    """
    Golub-Welsch algorithm for creating quadrature nodes and weights.

    Args:
        order (int) : Quadrature order
        dist (Dist) : Distribution nodes and weights are found for with
            `dim=len(dist)`
        accuracy (int) : Accuracy used in discretized Stieltjes procedure. Will
            be increased by one for each itteration.

    Returns:
        (numpy.array, numpy.array) : Optimal collocation nodes with
            `x.shape=(dim, order+1)` and weights with `w.shape=(order+1,)`.

    Examples:
        >>> Z = chaospy.Normal()
        >>> x, w = chaospy.quad_golub_welsch(3, Z)
        >>> print(x)
        [[-2.33441422 -0.74196378  0.74196378  2.33441422]]
        >>> print(w)
        [ 0.04587585  0.45412415  0.45412415  0.04587585]

        Multivariate
        >>> Z = chaospy.J(chaospy.Uniform(), chaospy.Uniform())
        >>> x, w = chaospy.quad_golub_welsch(1, Z)
        >>> print(x)
        [[ 0.21132487  0.21132487  0.78867513  0.78867513]
         [ 0.21132487  0.78867513  0.21132487  0.78867513]]
        >>> print(w)
        [ 0.25  0.25  0.25  0.25]
    """
    order = numpy.array(order)*numpy.ones(len(dist), dtype=int)+1
    _, _, coeff1, coeff2 = chaospy.quad.generate_stieltjes(
        dist, numpy.max(order), accuracy=accuracy, retall=True, **kws)

    dimensions = len(dist)
    abscisas, weights = _golbub_welsch(order, coeff1, coeff2)

    if dimensions == 1:
        abscisa = numpy.reshape(abscisas, (1, order[0]))
        weight = numpy.reshape(weights, (order[0],))
    else:
        abscisa = chaospy.quad.combine(abscisas).T
        weight = numpy.prod(chaospy.quad.combine(weights), -1)

    assert len(abscisa) == dimensions
    assert len(weight) == len(abscisa.T)
    return abscisa, weight


def _golbub_welsch(orders, coeff1, coeff2):
    """Recurrence coefficients to abscisas and weights."""
    abscisas, weights = [], []

    for dim, order in enumerate(orders):
        if order:
            bands = numpy.empty((2, order))
            bands[0] = coeff1[dim, :order]
            bands[1, :-1] = numpy.sqrt(coeff2[dim, 1:order])
            vals, vecs = scipy.linalg.eig_banded(bands, lower=True)

            abscisa, weight = vals.real, vecs[0, :]**2
            indices = numpy.argsort(abscisa)
            abscisa, weight = abscisa[indices], weight[indices]

        else:
            abscisa, weight = numpy.array([coeff1[dim, 0]]), numpy.array([1.])

        abscisas.append(abscisa)
        weights.append(weight)
    return abscisas, weights

if __name__ == "__main__":
    import doctest
    doctest.testmod()
