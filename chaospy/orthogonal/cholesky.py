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

APPROXIMATE_POSITIVE_DEFINITENES = {"none": lambda mat: mat}
try:
    import matrix
    APPROXIMATE_POSITIVE_DEFINITENES.update(
        reimer=matrix.approximation.positive_semidefinite.positive_semidefinite_matrix,
        gmw_81=matrix.approximation.positive_semidefinite.GMW_81,
        gmw_t1=matrix.approximation.positive_semidefinite.GMW_T1,
        gmw_t2=matrix.approximation.positive_semidefinite.GMW_T2,
        se_90=matrix.approximation.positive_semidefinite.SE_90,
        se_99=matrix.approximation.positive_semidefinite.SE_99,
        se_t1=matrix.approximation.positive_semidefinite.SE_T1,
    )
except ImportError:  # pragma: no coverage
    pass


def orth_chol(
    order,
    dist,
    normed=False,
    graded=True,
    reverse=True,
    cross_truncation=1.,
    retall=False,
    approx_method=None,
):
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
        approx_method (Optional[str]):
            Method to use in case the expansion covariance does not identify as
            positive definite. The methods are as follows:

            * ``none`` -- No approximation applied. Raises error instead.
            * ``gmw_81`` -- :func:`~matrix.approximation.positive_semidefinite.GMW_81`
            * ``gmw_t1`` -- :func:`~matrix.approximation.positive_semidefinite.GMW_T1`
            * ``gmw_t2`` -- :func:`~matrix.approximation.positive_semidefinite.GMW_T2`
            * ``se_90`` -- :func:`~matrix.approximation.positive_semidefinite.SE_90`
            * ``se_99`` -- :func:`~matrix.approximation.positive_semidefinite.SE_99`
            * ``se_t1`` -- :func:`~matrix.approximation.positive_semidefinite.SE_T1`
            * ``reimer`` -- :func:`~matrix.approximation.positive_semidefinite.positive_semidefinite_matrix`

            Defaults 'se_99' if matrix-decomposition is installed (which
            requires >=python3.7), 'none' otherwise.

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

    if approx_method is None:
        approx_method = "se99" if "se99" in APPROXIMATE_POSITIVE_DEFINITENES else "none"

    covariance = chaospy.descriptives.Cov(basis, dist)
    make_positive_definite = APPROXIMATE_POSITIVE_DEFINITENES[approx_method]
    covariance = make_positive_definite(covariance)
    cholmat = numpy.linalg.cholesky(covariance)
    # matrix.decompose(covariance, return_type="LL").L

    cholmat_inv = numpy.linalg.inv(cholmat.T).T
    if not normed:
        diag_mesh = numpy.repeat(numpy.diag(cholmat_inv), len(cholmat_inv))
        cholmat_inv /= diag_mesh.reshape(cholmat_inv.shape)
        norms = numpy.hstack([1, numpy.diag(cholmat)**2])
    else:
        norms = numpy.ones(length+1, dtype=float)

    expected = -numpy.sum(cholmat_inv*chaospy.E(basis, dist), -1)
    coeffs = numpy.block([[1.,                       expected],
                          [numpy.zeros((length, 1)), cholmat_inv.T]])

    out = {}
    out[(0,)*dim] = coeffs[0]
    for idx, key in enumerate(basis.exponents):
        out[tuple(key)] = coeffs[idx+1]

    names = numpoly.symbols("q:%d" % dim)
    polynomials = numpoly.polynomial(out, names=names)

    if retall:
        return polynomials, norms
    return polynomials
