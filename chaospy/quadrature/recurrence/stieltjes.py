"""Discretized Stieltjes' method."""
import numpy
import numpoly


def discretized_stieltjes(order, abscissas, weights, normed=False):
    """
    Discretized Stieltjes' method.

    Args:
        order (int):
            The polynomial order create.
        abscissas (numpy.ndarray):
            Quadrature abscissas, assumed to be of ``shape == (D, N)``, where
            ``D`` is the number of distributions to handle at once, and ``N``
            is the number of abscissas.
        weights (numpy.ndarray):
            Quadrature weights, assumed to be of ``shape == (N,)``.
        normed (bool):
            If true, normalize polynomials.

    Returns:
        (numpy.ndarray, numpoly.ndpoly, numpy.ndarray):
            coefficients:
                The recurrence coefficients created using the discretized
                Stieltjes' method, with ``shape == (2, D, order+1)``.
            polynomials:
                The orthogonal polynomial expansion created as a by-product of
                the algorithm.
            norms:
                The norm of each orthogonal polynomial. Roughly equivalent to
                ``chaospy.E(polynomials**2, dist)``, but more numerically
                stable than most alternatives.

    Examples:
        >>> dist = chaospy.J(chaospy.Uniform(0, 1), chaospy.Beta(3, 4))
        >>> abscissas, weights = chaospy.generate_quadrature(
        ...     9, dist, rule="clenshaw_curtis")
        >>> coeffs, orth, norms = discretized_stieltjes(2, abscissas, weights)
        >>> print(coeffs.round(5))
        [[[0.5     0.5     0.5    ]
          [0.42857 0.46032 0.47525]]
        <BLANKLINE>
         [[1.      0.08333 0.06667]
          [1.      0.03061 0.04321]]]
        >>> print(orth[2].round(5))
        [0.16667-q0+q0**2 0.16667-0.88889*q1+q1**2]
        >>> print(norms.round(5))
        [[1.      0.08333 0.00556]
         [1.      0.03061 0.00132]]
        >>> coeffs, orth, norms = discretized_stieltjes(
        ...     2, abscissas, weights, normed=True)
        >>> print(coeffs.round(5))
        [[[0.5     0.04167 0.26424]
          [0.42857 0.01409 0.31365]]
        <BLANKLINE>
         [[1.      1.      1.     ]
          [1.      1.      1.     ]]]
        >>> print(orth[2].round(5))
        [-1.04874-2.1209*q0+3.9155*q0**2 -1.00494-2.63343*q1+5.94906*q1**2]
        >>> print(norms.round(5))
        [[1. 1. 1.]
         [1. 1. 1.]]
    """
    abscissas = numpy.asfarray(abscissas)
    weights = numpy.asfarray(weights)
    assert len(weights.shape) == 1
    assert len(abscissas.shape) == 2
    assert abscissas.shape[-1] == len(weights)

    poly = numpoly.symbols("q:%d" % len(abscissas))
    orth = list(numpoly.polynomial([
        numpy.zeros(len(abscissas)), numpy.ones(len(abscissas))]))

    inner = numpy.sum(abscissas*weights, -1)
    norms = [numpy.ones(len(abscissas)), numpy.ones(len(abscissas))]
    coeffs = []

    for _ in range(int(order)):

        coeffs.append((inner/norms[-1], norms[-1]/norms[-2]))
        orth.append((poly-coeffs[-1][0])*orth[-1] - orth[-2]*coeffs[-1][1])

        raw_nodes = orth[-1](*abscissas)**2*weights
        inner = numpy.sum(abscissas*raw_nodes, -1)
        norms.append(numpy.sum(raw_nodes, -1))

        if normed:
            orth[-1] = orth[-1]/numpy.sqrt(norms[-1])
            norms[-1] **= 0

    coeffs.append((inner/norms[-1], norms[-1]/norms[-2]))

    coeffs = numpy.moveaxis(coeffs, 0, 2)
    norms = numpy.array(norms[1:]).T

    orth = [poly[numpy.newaxis] for poly in orth]
    orth = numpoly.concatenate(orth[1:], axis=0)

    return coeffs, orth, norms



def analytical_stieljes(order, dist, normed=False):
    """
    Examples:
        >>> dist = chaospy.J(chaospy.Uniform(0, 1), chaospy.Beta(3, 4))
        >>> coeffs, orth, norms = analytical_stieljes(2, dist)
        >>> print(coeffs.round(5))
        [[[0.5     0.5     0.5    ]
          [0.42857 0.46032 0.47475]]
        <BLANKLINE>
         [[1.      0.08333 0.06667]
          [1.      0.03061 0.04321]]]
        >>> print(orth[:, 2].round(5))
        [0.16667-q0+q0**2 0.16667-0.88889*q1+q1**2]
        >>> print(norms.round(5))
        [[1.      0.08333 0.00556]
         [1.      0.03061 0.00132]]
        >>> coeffs, orth, norms = analytical_stieljes(2, dist, normed=True)
        >>> print(coeffs.round(5))
        [[[0.5     0.5     0.5    ]
          [0.42857 0.46032 0.47475]]
        <BLANKLINE>
         [[1.      0.08333 0.06667]
          [1.      0.03061 0.04321]]]
        >>> print(orth[:, 2].round(5))
        [2.23607-13.41641*q0+13.41641*q0**2 4.58258-24.4404*q1+27.49545*q1**2]
        >>> print(norms.round(5))
        [[1. 1. 1.]
         [1. 1. 1.]]
    """
    dimensions = len(dist)
    mom_order = numpy.arange(order+1).repeat(dimensions)
    mom_order = mom_order.reshape(order+1, dimensions).T
    coeffs = dist.ttr(mom_order)
    coeffs[1, :, 0] = 1.

    var = numpoly.symbols("q:%d" % len(dist))
    orth = list(numpoly.polynomial([
        numpy.zeros((1, len(dist))), numpy.ones((1, len(dist)))]))

    for order_ in range(order):
        orth.append(
            orth[-1]*(var-coeffs[0, :, order_])
            - orth[-2]*coeffs[1, :, order_]
        )
    orth = numpoly.concatenate(orth[1:], axis=0).T

    norms = numpy.cumprod(coeffs[1], 1)
    if normed:
        orth = orth/numpy.sqrt(norms)
        norms **= 0

    return coeffs, orth, norms
