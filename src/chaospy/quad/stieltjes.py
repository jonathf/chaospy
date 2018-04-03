"""
Implementation of Stieltjes' method.
"""
import numpy

import chaospy.poly.base
import chaospy.poly.collection
import chaospy.quad
from .. import distributions


def generate_stieltjes(
        dist, order, accuracy=100, normed=False, retall=False, **kws):
    """
    Discretized Stieltjes' method.

    Args:
        dist (Dist):
            Distribution defining the space to create weights for.
        order (int):
            The polynomial order create.
        accuracy (int):
            The quadrature order of the Clenshaw-Curtis nodes to
            use at each step, if approximation is used.
        retall (bool):
            If included, more values are returned

    Returns:
        (list): List of polynomials, norms of polynomials and three terms
            coefficients. The list created from the method with
            ``len(orth) == order+1``. If ``len(dist) > 1``, then each
            polynomials are multivariate.
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): If ``retall`` is true,
            also return polynomial norms and the three term coefficients.
            The norms of the polynomials with ``norms.shape = (dim, order+1)``
            where ``dim`` are the number of dimensions in dist.  The
            coefficients have ``shape == (dim, order+1)``.

    Examples:
        >>> dist = chaospy.J(chaospy.Normal(), chaospy.Weibull())
        >>> orth, norms, coeffs1, coeffs2 = chaospy.generate_stieltjes(
        ...     dist, 2, retall=True)
        >>> print(chaospy.around(orth[2], 5))
        [q0^2-1.0, q1^2-4.0q1+2.0]
        >>> print(numpy.around(norms, 5))
        [[1. 1. 2.]
         [1. 1. 4.]]
        >>> print(numpy.around(coeffs1, 5))
        [[0. 0. 0.]
         [1. 3. 5.]]
        >>> print(numpy.around(coeffs2, 5))
        [[1. 1. 2.]
         [1. 1. 4.]]

        >>> dist = chaospy.Uniform()
        >>> orth, norms, coeffs1, coeffs2 = chaospy.generate_stieltjes(
        ...     dist, 2, retall=True)
        >>> print(chaospy.around(orth[2], 8))
        q0^2-q0+0.16666667
        >>> print(numpy.around(norms, 4))
        [[1.     0.0833 0.0056]]
    """
    assert not distributions.evaluation.get_dependencies(dist)

    if len(dist) > 1:

        # one for each dimension:
        orth, norms, coeff1, coeff2 = zip(*[generate_stieltjes(
            _, order, accuracy, normed, retall=True, **kws) for _ in dist])

        # ensure each polynomial has its own dimension:
        orth = [[chaospy.setdim(_, len(orth)) for _ in poly] for poly in orth]
        orth = [[chaospy.rolldim(_, len(dist)-idx) for _ in poly] for idx, poly in enumerate(orth)]
        orth = [chaospy.poly.base.Poly(_) for _ in zip(*orth)]

        if not retall:
            return orth

        # stack results:
        norms = numpy.vstack(norms)
        coeff1 = numpy.vstack(coeff1)
        coeff2 = numpy.vstack(coeff2)

        return orth, norms, coeff1, coeff2

    try:
        orth, norms, coeff1, coeff2 = _stieltjes_analytical(
            dist, order, normed)
    except NotImplementedError:
        orth, norms, coeff1, coeff2 = _stieltjes_approx(
            dist, order, accuracy, normed, **kws)

    if retall:
        return orth, norms, coeff1, coeff2
    return orth


def _stieltjes_analytical(dist, order, normed):
    """Stieltjes' method with analytical recurrence coefficients."""
    dimensions = len(dist)
    mom_order = numpy.arange(
        order+1
    ).repeat(
        dimensions
    ).reshape(
        order+1, dimensions
    ).T
    coeff1, coeff2 = dist.ttr(mom_order)
    coeff2[:, 0] = 1.

    poly = chaospy.poly.collection.core.variable(dimensions)
    if normed:
        orth = [
            poly**0*numpy.ones(dimensions),
            (poly-coeff1[:, 0])/numpy.sqrt(coeff2[:, 1]),
        ]
        for order_ in range(1, order):
            orth.append(
                (orth[-1]*(poly-coeff1[:, order_])
                 -orth[-2]*numpy.sqrt(coeff2[:, order_]))
                /numpy.sqrt(coeff2[:, order_+1])
            )
        norms = numpy.ones(coeff2.shape)
    else:
        orth = [poly-poly, poly**0*numpy.ones(dimensions)]
        for order_ in range(order):
            orth.append(
                orth[-1]*(poly-coeff1[:, order_])
                - orth[-2]*coeff2[:, order_]
            )
        orth = orth[1:]
        norms = numpy.cumprod(coeff2, 1)

    return orth, norms, coeff1, coeff2


def _stieltjes_approx(dist, order, accuracy, normed, **kws):
    """Stieltjes' method with approximative recurrence coefficients."""
    kws["rule"] = kws.get("rule", "C")
    assert kws["rule"].upper() != "G"

    absisas, weights = chaospy.quad.generate_quadrature(
        accuracy, dist.range(), **kws)
    weights = weights*dist.pdf(absisas)

    poly = chaospy.poly.variable(len(dist))
    orth = [poly*0, poly**0]

    inner = numpy.sum(absisas*weights, -1)
    norms = [numpy.ones(len(dist)), numpy.ones(len(dist))]
    coeff1 = []
    coeff2 = []

    for _ in range(order):

        coeff1.append(inner/norms[-1])
        coeff2.append(norms[-1]/norms[-2])
        orth.append((poly-coeff1[-1])*orth[-1] - orth[-2]*coeff2[-1])

        raw_nodes = orth[-1](*absisas)**2*weights
        inner = numpy.sum(absisas*raw_nodes, -1)
        norms.append(numpy.sum(raw_nodes, -1))

        if normed:
            orth[-1] = orth[-1]/numpy.sqrt(norms[-1])

    coeff1.append(inner/norms[-1])
    coeff2.append(norms[-1]/norms[-2])
    coeff1 = numpy.transpose(coeff1)
    coeff2 = numpy.transpose(coeff2)
    norms = numpy.array(norms[1:]).T
    orth = orth[1:]
    return orth, norms, coeff1, coeff2
