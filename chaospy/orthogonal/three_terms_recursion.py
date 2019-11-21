r"""
Creating an orthogonal polynomial expansion can be numerical unstable
when using raw statistical moments as input (see `paper by Gautschi`_ for
details). This can be a problem if constructing large expansions since the
error blows up. Given that the distribution is univariate it is instead
possible to create orthogonal polynomials stabilized using the three terms
recursion relation:

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

In the ``chaospy`` toolbox three terms recursion coefficient can be
generating by calling the ``ttr`` instance method::

    >>> dist = chaospy.Uniform(-1,1)
    >>> print(numpy.around(dist.ttr([0,1,2,3]), 4))
    [[ 0.      0.      0.      0.    ]
     [-0.      0.3333  0.2667  0.2571]]

In many of the pre-defined probability distributions in ``chaospy``, the three
terms recursion coefficients are calculated analytically. If the distribution
does not support the method, the coefficients are instead calculated using the
discretized Stieltjes method (described in the `paper by Golub and Welsch`_).

In ``chaospy`` constructing orthogonal polynomial using the three term
recursion scheme can be done through ``orth_ttr``. For example::

    >>> dist = chaospy.Iid(chaospy.Gamma(1), 2)
    >>> orths = chaospy.orth_ttr(2, dist)
    >>> orths.round(4)
    polynomial([1.0, -1.0+q1, -1.0+q0, 2.0-4.0*q1+q1**2, 1.0-q1-q0+q0*q1,
                2.0-4.0*q0+q0**2])

The method will use the ``ttr`` function if available, and discretized
Stieltjes otherwise.

.. _paper by Gautschi: https://www.ams.org/journals/mcom/1968-22-102/S0025-5718-1968-0228171-0/
.. _paper by Golub and Welsch: https://web.stanford.edu/class/cme335/spr11/S0025-5718-69-99647-1.pdf
"""
import numpy
import chaospy


def orth_ttr(
        order, dist, normed=False, sort="G", retall=False,
        cross_truncation=1., **kws):
    """
    Create orthogonal polynomial expansion from three terms recursion formula.

    Args:
        order (int):
            Order of polynomial expansion.
        dist (Dist):
            Distribution space where polynomials are orthogonal If dist.ttr
            exists, it will be used. Must be stochastically independent.
        normed (bool):
            If True orthonormal polynomials will be used.
        sort (str):
            Polynomial sorting. Same as in basis.
        retall (bool):
            If true return numerical stabilized norms as well. Roughly the same
            as ``cp.E(orth**2, dist)``.
        cross_truncation (float):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion. only include terms where the exponents ``K``
            satisfied the equation
            ``order >= sum(K**(1/cross_truncation))**cross_truncation``.

    Returns:
        (chaospy.poly.ndpoly, numpy.ndarray):
            Orthogonal polynomial expansion and norms of the orthogonal
            expansion on the form ``E(orth**2, dist)``. Calculated using
            recurrence coefficients for stability.

    Examples:
        >>> Z = chaospy.Normal()
        >>> chaospy.orth_ttr(4, Z).round(4)
        polynomial([1.0, q0, -1.0+q0**2, -3.0*q0+q0**3, 3.0-6.0*q0**2+q0**4])
    """
    try:
        _, polynomials, norms, = chaospy.quadrature.recurrence.analytical_stieljes(
            numpy.max(order), dist, normed=normed)
    except NotImplementedError:
        abscissas, weights = chaospy.quadrature.generate_quadrature(
            int(10000**(1/len(dist))), dist, rule="F")
        _, polynomials, norms, = chaospy.quadrature.recurrence.discretized_stieltjes(
            numpy.max(order), abscissas, weights, normed=normed)

    polynomials = polynomials.reshape((len(dist), numpy.max(order)+1))

    indices = chaospy.bertran.bindex(
        start=0, stop=order, dim=len(dist), sort=sort,
        cross_truncation=cross_truncation,
    )
    if len(dist) > 1:
        polynomials = chaospy.poly.prod(chaospy.polynomial([
            poly[idx] for poly, idx in zip(polynomials, indices.T)]), 0)
        norms = numpy.prod([
            norms_[idx] for norms_, idx in zip(norms, indices.T)], 0)
    else:
        polynomials = polynomials.flatten()

    if retall:
        return polynomials, norms
    return polynomials
