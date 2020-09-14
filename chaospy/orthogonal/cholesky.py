"""
Even though orthogonal polynomials created using three terms recurrence is
the recommended approach as it is the most numerical stable method, it can not
be used directly on stochastically dependent random variables. On method that
bypasses this problem is Cholesky decomposition method.

Cholesky exploits the fact that except for polynomials that are constant, there
is a equivalence between orthogonality and uncorrelated. Decorrelation can be
achieved in a simple few steps:

1. Start with any linear independent basis of polynomials :math:`P`.
2. Temporarily remove any constants, if there are any.
3. Construct the covariance matrix :math`C=Cov(P)` of all polynomials.
4. Decompose covariance matrix into two parts :math:`C = L^T L` using Cholesky
   decomposition.
5. Multiply the vector of polynomials with the inverse decomposition:
   :math:`Q = P L^{-1}`
6. If a constant was removed, subtract the mean from the vector :math`Q=Q-E[Q]`
   before adding the constant back into the expansion.

This should work in theory, but in practice step 4 is known to be numerically
unstable. To this end, this step can to a certain degree be regularized using
various methods. To this end a few modified Cholesky decompositions are
available in ``chaospy``.
"""
import numpy
import chaospy
import numpoly


def orth_chol(order, dist, normed=False, graded=True, reverse=True,
              cross_truncation=1., retall=False, **kws):
    """
    Create orthogonal polynomial expansion from Cholesky decomposition.

    Args:
        order (int):
            Order of polynomial expansion
        dist (Distribution):
            Distribution space where polynomials are orthogonal
        normed (bool):
            If True orthonormal polynomials will be used instead of monic.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``q0**2*q1*q2``,
            ``q0*q1**2*q2`` and ``q0*q1*q2**2``, which all have exponent sum of
            5.
        reverse (bool):
            Reverse lexicographical sorting meaning that ``q0*q1**3`` is
            considered bigger than ``q0**3*q1``, instead of the opposite.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.
        retall (bool):
            If true return numerical stabilized norms as well. Roughly the same
            as ``cp.E(orth**2, dist)``.

    Examples:
        >>> distribution = chaospy.Normal()
        >>> expansion, norms = chaospy.orth_chol(3, distribution, retall=True)
        >>> expansion.round(4)
        polynomial([1.0, q0, q0**2-1.0, q0**3-3.0*q0])
        >>> norms
        array([1., 1., 2., 6.])

    """
    dim = len(dist)
    basis = numpoly.monomial(
        start=1,
        stop=order+1,
        names=numpoly.variable(dim).names,
        graded=graded,
        reverse=reverse,
        cross_truncation=cross_truncation,
    )
    length = len(basis)

    covariance = chaospy.descriptives.Cov(basis, dist)
    cholmat = chaospy.chol.gill_king(covariance)
    cholmat_inv = numpy.linalg.inv(cholmat.T).T
    if not normed:
        diag_mesh = numpy.repeat(numpy.diag(cholmat_inv), len(cholmat_inv))
        cholmat_inv /= diag_mesh.reshape(cholmat_inv.shape)
        norms = numpy.hstack([1, numpy.diag(cholmat)**2])
    else:
        norms = numpy.ones(length+1, dtype=float)

    coefs = numpy.empty((length+1, length+1))

    coefs[1:, 1:] = cholmat_inv
    coefs[0, 0] = 1
    coefs[0, 1:] = 0

    expected = -numpy.sum(
        cholmat_inv*chaospy.descriptives.E(basis, dist, **kws), -1)
    coefs[1:, 0] = expected

    coefs = coefs.T

    out = {}
    out[(0,)*dim] = coefs[0]
    for idx, key in enumerate(basis.exponents):
        out[tuple(key)] = coefs[idx+1]

    names = numpoly.symbols("q:%d" % dim)
    polynomials = numpoly.polynomial(out, names=names)

    if retall:
        return polynomials, norms
    return polynomials
