"""

"""
import numpy
import numpoly


def fit_quadrature(
        orth,
        nodes,
        weights,
        solves,
        retall=False,
        norms=None
):
    """
    Fit polynomial chaos expansion using spectral projection.

    Create a polynomial approximation model from orthogonal expansion,
    quadrature nodes and weights.

    Args:
        orth (numpoly.ndpoly):
            Orthogonal polynomial expansion. Must be orthogonal for the
            approximation to be accurate.
        nodes (numpy.ndarray):
            Where to evaluate the polynomial expansion and model to
            approximate. ``nodes.shape==(D, K)`` where ``D`` is the number of
            dimensions and ``K`` is the number of nodes.
        weights (numpy.ndarray):
            Weights when doing numerical integration. ``weights.shape == (K,)``
            must hold.
        solves (numpy.ndarray):
            The model evaluation to approximate. If `numpy.ndarray` is
            provided, it must have ``len(solves) == K``.
        retall (int):
            What the function should return.
            0: only return fitted polynomials, with shape `evals.shape[1:]`.
            1: polynomials, and Fourier coefficients,
            2: polynomials, coefficients and polynomial evaluations.
        norms (numpy.ndarray):
            Three terms recurrence method produces norms more stable than the
            ones calculated from the polynomials themselves. Calculated from
            quadrature if not provided. ``norms.shape == (len(orth),)`` must
            hold.

    Returns:
        (numpoly.ndpoly):
            Fitted model approximation in the form of an polynomial.
    """
    orth = numpoly.polynomial(orth)
    assert orth.ndim == 1
    weights = numpy.asfarray(weights)
    assert weights.ndim == 1
    solves = numpy.asfarray(solves)
    nodes = numpy.atleast_2d(nodes)
    assert nodes.ndim == 2
    assert nodes.shape[1] == len(weights) == len(solves)

    shape = solves.shape[1:]
    solves = solves.reshape(len(solves), -1)

    ovals = orth(*nodes)
    vals1 = [(val*solves.T*weights).T for val in ovals]

    if norms is None:
        norms = numpy.sum(ovals**2*weights, -1)
    norms = numpy.asfarray(norms)
    assert norms.ndim == 1

    coeffs = (numpy.sum(vals1, 1).T/norms).T
    coeffs = coeffs.reshape(len(coeffs), *shape)
    approx_model = numpoly.sum(orth*coeffs.T, -1).T

    choices = {0: approx_model,
               1: (approx_model, coeffs),
               2: (approx_model, coeffs, ovals)}
    return choices[retall]
