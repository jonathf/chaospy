r"""
Creating an orthogonal polynomial expansion can be numerical unstable
when using raw statistical moments as input (see `paper by Gautschi`_ for
details). This can be a problem if constructing large expansions since the
error blows up. Given that the distribution is univariate it is instead
possible to create orthogonal polynomials stabilized using the three terms
recurrence relation:

.. math::
    \Phi_{n+1}(q) = \Phi_{n}(q) (q-A_n) - \Phi_{n-1}(q) B_n,

where

.. math::
    A_n = \frac{\left\langle q \Phi_n, \Phi_n \right\rangle}{
          \left\langle \Phi_n, \Phi_n \right\rangle}
        = \frac{\mathbb E[q\Phi_{n}^2]}{
          \mathbb E[\Phi_n^2]}
    B_n = \frac{\left\langle \Phi_n, \Phi_n \right\rangle}{
          \left\langle \Phi_{n-1}, \Phi_{n-1} \right\rangle}
        = \frac{\mathbb E[\Phi_{n}^2]}{\mathbb E[\Phi_{n-1}^2]}

A multivariate polynomial expansion can be created using tensor product
rule of univariate polynomials expansions. This assumes that the
distribution is stochastically independent.

In the ``chaospy`` toolbox three terms recurrence coefficient can be
generating by calling the ``ttr`` instance method::

    >>> dist = chaospy.Uniform(-1, 1)
    >>> dist.ttr([0,1,2,3]).round(4)
    array([[ 0.    ,  0.    ,  0.    ,  0.    ],
           [-0.    ,  0.3333,  0.2667,  0.2571]])

In many of the pre-defined probability distributions in ``chaospy``, the three
terms recurrence coefficients are calculated analytically. If the distribution
does not support the method, the coefficients are instead calculated using the
discretized Stieltjes method (described in the `paper by Golub and Welsch`_).

In ``chaospy`` constructing orthogonal polynomial using the three term
recurrence scheme can be done through ``orth_ttr``. For example::

    >>> dist = chaospy.Iid(chaospy.Gamma(1), 2)
    >>> orths = chaospy.orth_ttr(2, dist)
    >>> orths.round(4)
    polynomial([1.0, q1-1.0, q0-1.0, q1**2-4.0*q1+2.0, q0*q1-q1-q0+1.0,
                q0**2-4.0*q0+2.0])

The method will use the ``ttr`` function if available, and discretized
Stieltjes otherwise.

.. _paper by Gautschi: https://www.ams.org/journals/mcom/1968-22-102/S0025-5718-1968-0228171-0/
.. _paper by Golub and Welsch: https://web.stanford.edu/class/cme335/spr11/S0025-5718-69-99647-1.pdf
"""
import logging

import numpy
import numpoly
import chaospy


def orth_ttr(order, dist, normed=False, graded=True, reverse=True,
             retall=False, cross_truncation=1., sort=None, **kws):
    """
    Create orthogonal polynomial expansion from three terms recurrence formula.

    Args:
        order (int):
            Order of polynomial expansion.
        dist (Distribution):
            Distribution space where polynomials are orthogonal If dist.ttr
            exists, it will be used. Must be stochastically independent.
        normed (bool):
            If True orthonormal polynomials will be used.
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
            terms in expansion. only include terms where the exponents ``K``
            satisfied the equation
            ``order >= sum(K**(1/cross_truncation))**cross_truncation``.

    Returns:
        (numpoly.ndpoly, numpy.ndarray):
            Orthogonal polynomial expansion. Norms of the orthogonal
            expansion on the form ``E(orth**2, dist)``. Calculated using
            recurrence coefficients for stability.

    Examples:
        >>> distribution = chaospy.J(chaospy.Normal(), chaospy.Normal())
        >>> polynomials, norms = chaospy.orth_ttr(2, distribution, retall=True)
        >>> polynomials.round(10)
        polynomial([1.0, q1, q0, q1**2-1.0, q0*q1, q0**2-1.0])
        >>> norms.round(10)
        array([1., 1., 1., 2., 1., 2.])
        >>> polynomials = chaospy.orth_ttr(2, distribution, normed=True)
        >>> polynomials.round(3)
        polynomial([1.0, q1, q0, 0.707*q1**2-0.707, q0*q1, 0.707*q0**2-0.707])

    """
    logger = logging.getLogger(__name__)
    if sort is not None:
        logger.warning("deprecation warning: 'sort' argument is deprecated; "
                       "use 'graded' and/or 'reverse' instead")
        graded = "G" in sort.upper()
        reverse = "R" not in sort.upper()

    try:
        _, polynomials, norms, = chaospy.quadrature.recurrence.analytical_stieljes(
            numpy.max(order), dist, normed=normed)
    except NotImplementedError:
        abscissas, weights = chaospy.quadrature.generate_quadrature(
            int(10000**(1/len(dist))), dist, rule="fejer")
        _, polynomials, norms, = chaospy.quadrature.recurrence.discretized_stieltjes(
            numpy.max(order), abscissas, weights, normed=normed)

    polynomials = polynomials.reshape((len(dist), numpy.max(order)+1))

    order = numpy.array(order)
    indices = numpoly.glexindex(start=0, stop=order+1, dimensions=len(dist),
                                graded=graded, reverse=reverse,
                                cross_truncation=cross_truncation)
    if len(dist) > 1:
        polynomials = numpoly.prod(chaospy.polynomial([
            poly[idx] for poly, idx in zip(polynomials, indices.T)]), 0)
        norms = numpy.prod([
            norms_[idx] for norms_, idx in zip(norms, indices.T)], 0)
    else:
        polynomials = polynomials.flatten()
        norms = norms.flatten()

    if retall:
        return polynomials, norms
    return polynomials
