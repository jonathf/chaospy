r"""
Creating an orthogonal polynomial expansion can be numerical unstable
when using raw statistical moments as input
:cite:`gautschi_construction_1968`. This can be a problem if
constructing large expansions since the error blows up. Given that the
distribution is univariate it is instead possible to create orthogonal
polynomials stabilized using the three terms recursion relation:

.. math::

    \Phi_{n+1}(q) = \Phi_{n}(q)(q-A_n) - \Phi_{n-1}(q) B_n,

where

.. math::

    A_n &= \frac{
        \left\langle
            q \Phi_n, \Phi_n
        \right\rangle
    }{
        \left\langle
            \Phi_n, \Phi_n
        \right\rangle
    } = \frac{
        \mathbb E[q\Phi_{n}^2]
    }{
        \mathbb E[\Phi_n^2]
    }

    B_n &= \frac{
        \left\langle
            \Phi_n, \Phi_n
        \right\rangle
    }{
        \left\langle
            \Phi_{n-1}, \Phi_{n-1}
        \right\rangle
    } = \frac{
        \mathbb E[\Phi_{n}^2]
    }{
        \mathbb E[\Phi_{n-1}^2]
    }

A multivariate polynomial expansion can be created using tensor product
rule of univariate polynomials expansions. This assumes that the
distribution is stochastically independent.

In the ``chaospy`` toolbox three terms recursion coefficient can be
generating by calling the ``ttr`` instance method::

    >>> dist = chaospy.Uniform(-1,1)
    >>> print(numpy.around(dist.ttr([0,1,2,3]), 4))
    [[ 0.      0.      0.      0.    ]
     [-0.      0.3333  0.2667  0.2571]]

Looking back to section :ref:`distributions` and the ``pc.contruct`` function,
``ttr`` can be added as a keyword argument. So tailored recursion coefficients
can be added for user defined distributions. If the ``ttr`` function is
missing, which can often the case, the coefficients can be calculated using
discretized Stieltjes method :cite:`gautschi_construction_1968`. The method
consists of alternating between calculating expression and expression given
starting values :math:`\Phi_{-1}=0` and :math:`\Phi_{0}=1`. Since the expected
value operator is used, this method can also be considered as a statistical
moment based method, however the coefficients calculations in equation can be
estimated using numerical integration, and made stable. In ``chaospy`` if the
``ttr`` is missing, it is estimated using ``quadgen`` with Clenshaw-Curtis
nodes and weights. The default is order 40, however, as with all the other
instance methods so far, it is
possible to set the wanted parameters using keyword argument. In this
case the keyword argument ``acc`` can be used to change the default. In
section :ref:`moments` the ``moment_generator`` function was introduced.
Analogous there is also a ``recurrence_generator`` function that does the same,
but for the ``ttr``. In other words, it is possible to fix the parameters in
the estimation of ``ttr`` in any distribution. Note that the keyword
``rule="G"`` is disabled since the Golub-Welsch algorithm also depends
upon the three terms recursion coefficients for it's calculations
:cite:`golub_calculation_1967`.

Multivariate orthogonal polynomial expansions are created by multiplying
univariate polynomials together:

.. math::

    \Phi_n = \Phi_{1,n_1}\cdots\Phi_{N,n_N}

where :math:`\Phi_{i,n_i}` represents the :math:`n_i`-th polynomial in
the univariate expansion orthogonal with respect to the :math:`i`-th
component of :math:`Q`. For the orthogonality to hold, it must be
assumed that :math:`p_{Q}` is stochastically independent. This to
assure the third equality in

.. math::

    \left\langle \Phi_n, \Phi_m \right\rangle &=
    \mathbb E[\Phi_n \Phi_m ] =
    \mathbb E[ \Phi_{1,n_1}\Phi_{i,m_1} \cdots\Phi_{N,n_N}
    \Phi_{N,m_N} ]

    &= \mathbb E[\Phi_{1,n_1}\Phi_{1,m_1}]\cdots
    \mathbb E[ \Phi_{N,n_N}\Phi_{N,m_N} ]

    &= \left\langle{\Phi_{1,n_1},\Phi_{1,m_1}}\right\rangle
    \cdots \left\langle{\Phi_{N,n_N},\Phi_{N,m_N}}\right\rangle.

Since each univariate polynomial expansion is orthogonal, this implies
that the multivariate also is orthogonal.

In ``chaospy`` constructing orthogonal polynomial using the three term
recursion scheme can be done through ``orth_ttr``. For example::

    >>> dist = chaospy.Iid(chaospy.Gamma(1), 2)
    >>> orths = chaospy.orth_ttr(2, dist)
    >>> print(chaospy.around(orths, 4))
    [1.0, q1-1.0, q0-1.0, q1^2-4.0q1+2.0, q0q1-q0-q1+1.0, q0^2-4.0q0+2.0]

The method will use the ``ttr`` function if available, and discretized
Stieltjes otherwise.
"""
import logging

import numpy
import chaospy

__all__ = (
    "orth_gs",
    "orth_ttr",
    "orth_chol",
    "norm",
)


def orth_gs(order, dist, normed=False, sort="GR", cross_truncation=1., **kws):
    """
    Gram-Schmidt process for generating orthogonal polynomials.

    Args:
        order (int, Poly):
            The upper polynomial order. Alternative a custom polynomial basis
            can be used.
        dist (Dist):
            Weighting distribution(s) defining orthogonality.
        normed (bool):
            If True orthonormal polynomials will be used instead of monic.
        sort (str):
            Ordering argument passed to poly.basis. If custom basis is used,
            argument is ignored.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.
        kws (optional):
            Keyword argument passed to dist.mom if the moments need to be
            estimated.

    Returns:
        (Poly) : The orthogonal polynomial expansion.

    Examples:
        >>> Z = chaospy.J(chaospy.Normal(), chaospy.Normal())
        >>> print(chaospy.around(chaospy.orth_gs(2, Z), 4))
        [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]
    """
    logger = logging.getLogger(__name__)
    dim = len(dist)

    if isinstance(order, int):
        if order == 0:
            return chaospy.poly.Poly(1, dim=dim)
        basis = chaospy.poly.basis(
            0, order, dim, sort, cross_truncation=cross_truncation)
    else:
        basis = order

    basis = list(basis)

    polynomials = [basis[0]]

    if normed:
        for idx in range(1, len(basis)):

            # orthogonalize polynomial:
            for idy in range(idx):
                orth = chaospy.descriptives.E(
                    basis[idx]*polynomials[idy], dist, **kws)
                basis[idx] = basis[idx] - polynomials[idy]*orth

            # normalize:
            norms = chaospy.descriptives.E(polynomials[-1]**2, dist, **kws)
            if norms <= 0:
                logger.warning("Warning: Polynomial cutoff at term %d", idx)
                break
            basis[idx] = basis[idx] / numpy.sqrt(norms)

            polynomials.append(basis[idx])

    else:

        norms = [1.]
        for idx in range(1, len(basis)):

            # orthogonalize polynomial:
            for idy in range(idx):
                orth = chaospy.descriptives.E(
                    basis[idx]*polynomials[idy], dist, **kws)
                basis[idx] = basis[idx] - polynomials[idy] * orth / norms[idy]

            norms.append(
                chaospy.descriptives.E(polynomials[-1]**2, dist, **kws))
            if norms[-1] <= 0:
                logger.warning("Warning: Polynomial cutoff at term %d", idx)
                break

            polynomials.append(basis[idx])

    return chaospy.poly.Poly(polynomials, dim=dim, shape=(len(polynomials),))


def orth_ttr(
        order, dist, normed=False, sort="GR", retall=False,
        cross_truncation=1., **kws):
    """
    Create orthogonal polynomial expansion from three terms recursion formula.

    Args:
        order (int):
            Order of polynomial expansion.
        dist (Dist):
            Distribution space where polynomials are orthogonal If dist.ttr
            exists, it will be used, othervice Clenshaw-Curtis integration will
            be used.  Must be stochastically independent.
        normed (bool):
            If True orthonormal polynomials will be used instead of monic.
        sort (str):
            Polynomial sorting. Same as in basis.
        retall (bool):
            If true return norms as well.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.
        kws (optional):
            Keyword argument passed to stieltjes method.

    Returns:
        orth (Poly, numpy.ndarray):
            Orthogonal polynomial expansion and norms of the orthogonal
            expansion on the form E(orth**2, dist). Calculated using recurrence
            coefficients for stability.

    Examples:
        >>> Z = chaospy.Normal()
        >>> print(chaospy.around(chaospy.orth_ttr(4, Z), 4))
        [1.0, q0, q0^2-1.0, q0^3-3.0q0, q0^4-6.0q0^2+3.0]
    """
    polynomials, norms, _, _ = chaospy.quad.generate_stieltjes(
        dist=dist, order=order, retall=True, **kws)

    if normed:
        for idx, poly in enumerate(polynomials):
            polynomials[idx] = poly / numpy.sqrt(norms[:, idx])
        norms = norms**0

    dim = len(dist)
    if dim > 1:
        mv_polynomials = []
        mv_norms = []
        indices = chaospy.bertran.bindex(
            start=0, stop=order, dim=dim, sort=sort,
            cross_truncation=cross_truncation,
        )

        for index in indices:
            poly = polynomials[index[0]][0]
            for idx in range(1, dim):
                poly = poly * polynomials[index[idx]][idx]
            mv_polynomials.append(poly)

        if retall:
            for index in indices:
                mv_norms.append(
                    numpy.prod([norms[idx, index[idx]] for idx in range(dim)]))

    else:
        mv_norms = norms[0]
        mv_polynomials = polynomials

    polynomials = chaospy.poly.flatten(chaospy.poly.Poly(mv_polynomials))

    if retall:
        return polynomials, numpy.array(mv_norms)
    return polynomials


def orth_chol(order, dist, normed=True, sort="GR", cross_truncation=1., **kws):
    """
    Create orthogonal polynomial expansion from Cholesky decomposition.

    Args:
        order (int) : Order of polynomial expansion
        dist (Dist) : Distribution space where polynomials are orthogonal
        normed (bool) : If True orthonormal polynomials will be used instead
                of monic.
        sort (str) : Ordering argument passed to poly.basis.  If custom basis
                is used, argument is ignored.
        cross_truncation (float) : Use hyperbolic cross truncation scheme to
                reduce the number of terms in expansion.
        kws (optional) : Keyword argument passed to dist.mom.

    Examples:
        >>> Z = chaospy.Normal()
        >>> print(chaospy.around(chaospy.orth_chol(3, Z), 4))
        [1.0, q0, 0.7071q0^2-0.7071, 0.4082q0^3-1.2247q0]
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
    for idx in range(length):
        index = basis[idx].keys[0]
        out[index] = coefs[idx+1]

    polynomials = chaospy.poly.Poly(out, dim, coefs.shape[1:], float)

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


def lagrange_polynomial(absicas, sort="GR"):
    """
    Create Lagrange polynomials.

    Args:
        absicas (array_like):
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
    absicas = numpy.asfarray(absicas)
    if len(absicas.shape) == 1:
        absicas = absicas.reshape(1, absicas.size)
    dim, size = absicas.shape

    order = 1
    while chaospy.bertran.terms(order, dim) <= size:
        order += 1

    indices = numpy.array(chaospy.bertran.bindex(0, order-1, dim, sort)[:size])
    idx, idy = numpy.mgrid[:size, :size]

    matrix = numpy.prod(absicas.T[idx]**indices[idy], -1)
    det = numpy.linalg.det(matrix)
    if det == 0:
        raise numpy.linalg.LinAlgError("invertible matrix")

    vec = chaospy.poly.basis(0, order-1, dim, sort)[:size]

    coeffs = numpy.zeros((size, size))

    if size == 1:
        out = chaospy.poly.basis(0, 0, dim, sort)*absicas.item()

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
