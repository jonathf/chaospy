"""Gram-Schmidt process for generating orthogonal polynomials."""
import logging

import numpy
import numpoly
import chaospy


def gram_schmidt(order, dist, normed=False, graded=True, reverse=True,
            retall=False, cross_truncation=1., **kws):
    """
    Gram-Schmidt process for generating orthogonal polynomials.

    Args:
        order (int, numpoly.ndpoly):
            The upper polynomial order. Alternative a custom polynomial basis
            can be used.
        dist (Distribution):
            Weighting distribution(s) defining orthogonality.
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
        retall (bool):
            If true return numerical stabilized norms as well. Roughly the same
            as ``cp.E(orth**2, dist)``.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.

    Returns:
        (chapspy.poly.ndpoly):
            The orthogonal polynomial expansion.

    Examples:
        >>> distribution = chaospy.J(chaospy.Normal(), chaospy.Normal())
        >>> polynomials, norms = chaospy.expansion.gram_schmidt(2, distribution, retall=True)
        >>> polynomials.round(10)
        polynomial([1.0, q1, q0, q1**2-1.0, q0*q1, q0**2-1.0])
        >>> norms.round(10)
        array([1., 1., 1., 2., 1., 2.])
        >>> polynomials = chaospy.expansion.gram_schmidt(2, distribution, normed=True)
        >>> polynomials.round(3)
        polynomial([1.0, q1, q0, 0.707*q1**2-0.707, q0*q1, 0.707*q0**2-0.707])

    """
    logger = logging.getLogger(__name__)
    dim = len(dist)

    if isinstance(order, int):
        order = numpoly.monomial(
            0,
            order+1,
            dimensions=numpoly.variable(2).names,
            graded=graded,
            reverse=reverse,
            cross_truncation=cross_truncation,
        )
    basis = list(order)
    polynomials = [basis[0]]

    norms = [1.]
    for idx in range(1, len(basis)):

        # orthogonalize polynomial:
        for idy in range(idx):
            orth = chaospy.E(basis[idx]*polynomials[idy], dist, **kws)
            basis[idx] = basis[idx]-polynomials[idy]*orth/norms[idy]

        norms_ = chaospy.E(basis[idx]**2, dist, **kws)
        if norms_ <= 0:  # pragma: no cover
            logger.warning("Warning: Polynomial cutoff at term %d", idx)
            break

        norms.append(1. if normed else norms_)
        basis[idx] = basis[idx]/numpy.sqrt(norms_) if normed else basis[idx]
        polynomials.append(basis[idx])

    polynomials = chaospy.polynomial(polynomials).flatten()
    if retall:
        norms = numpy.array(norms)
        return polynomials, norms
    return polynomials
