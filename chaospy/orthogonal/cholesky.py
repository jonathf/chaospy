"""
Even though orthogonal polynomials created using three terms recursion is
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


def orth_chol(order, dist, normed=True, sort="G", cross_truncation=1., **kws):
    """
    Create orthogonal polynomial expansion from Cholesky decomposition.

    Args:
        order (int):
            Order of polynomial expansion
        dist (Dist):
            Distribution space where polynomials are orthogonal
        normed (bool):
            If True orthonormal polynomials will be used instead of monic.
        sort (str):
            Ordering argument passed to poly.basis.  If custom basis is used,
            argument is ignored.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.

    Examples:
        >>> Z = chaospy.Normal()
        >>> chaospy.orth_chol(3, Z).round(4)
        polynomial([1.0, q0, -0.7071+0.7071*q0**2, -1.2247*q0+0.4082*q0**3])
    """
    dim = len(dist)
    basis = chaospy.poly.basis(
        start=1, stop=order, dim=dim, sort=sort,
        cross_truncation=cross_truncation,
    )
    length = len(basis)

    cholmat = chaospy.chol.gill_king(chaospy.descriptives.Cov(basis, dist))
    cholmat_inv = numpy.linalg.inv(cholmat.T).T
    if not normed:
        diag_mesh = numpy.repeat(numpy.diag(cholmat_inv), len(cholmat_inv))
        cholmat_inv /= diag_mesh.reshape(cholmat_inv.shape)

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
    polynomials = chaospy.poly.polynomial(out, names=names)

    return polynomials


def norm(order, dist, orth=None):

    dim = len(dist)
    try:
        if dim>1:
            norms = numpy.array([norm(order+1, D) for D in dist])
            Is = chaospy.bertran.bindex(order, dim)
            out = numpy.ones(len(Is))

            for i in range(len(Is)):
                index = Is[i]
                for j in range(dim):
                    if index[j]:
                        out[i] *= norms[j, index[j]]
            return out

        K = range(1,order+1)
        ttr = [1.] + [dist.ttr(k)[1] for k in K]
        return numpy.cumprod(ttr)

    except NotImplementedError:

        if orth is None:
            orth = orth_chol(order, dist)
        return chaospy.descriptives.E(orth**2, dist)
