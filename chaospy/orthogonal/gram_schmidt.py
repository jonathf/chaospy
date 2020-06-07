"""
Perhaps the simplest method for creating orthogonal polynomials is
Gram-Schmidt's method. It consist of using projections to iteratively make new
terms in an expansion orthogonal to the previous ones.
"""
import logging

import numpy
import chaospy


def orth_gs(order, dist, normed=False, graded=True, reverse=True,
            cross_truncation=1., sort=None, **kws):
    """
    Gram-Schmidt process for generating orthogonal polynomials.

    Args:
        order (int, chaospy.poly.ndpoly):
            The upper polynomial order. Alternative a custom polynomial basis
            can be used.
        dist (Dist):
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
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion.

    Returns:
        (chapspy.poly.ndpoly):
            The orthogonal polynomial expansion.

    Examples:
        >>> Z = chaospy.J(chaospy.Normal(), chaospy.Normal())
        >>> chaospy.orth_gs(2, Z).round(4)
        polynomial([1.0, q1, q0, q1**2-1.0, q0*q1, q0**2-1.0])
    """
    logger = logging.getLogger(__name__)
    dim = len(dist)

    if isinstance(order, int):
        if order == 0:
            return chaospy.poly.polynomial(1, names=("q0",))
        basis = chaospy.poly.basis(0, order, dim, graded=graded, reverse=reverse,
                                   sort=sort, cross_truncation=cross_truncation)
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

    return chaospy.poly.polynomial(polynomials).reshape((len(polynomials),))
