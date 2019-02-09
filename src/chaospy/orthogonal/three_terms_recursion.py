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
    >>> print(chaospy.around(orths, 4))
    [1.0, q1-1.0, q0-1.0, q1^2-4.0q1+2.0, q0q1-q0-q1+1.0, q0^2-4.0q0+2.0]

The method will use the ``ttr`` function if available, and discretized
Stieltjes otherwise.

.. _paper by Gautschi: https://www.ams.org/journals/mcom/1968-22-102/S0025-5718-1968-0228171-0/
.. _paper by Golub and Welsch: https://web.stanford.edu/class/cme335/spr11/S0025-5718-69-99647-1.pdf
"""
import numpy
import chaospy


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
        (Poly, numpy.ndarray):
            Orthogonal polynomial expansion and norms of the orthogonal
            expansion on the form ``E(orth**2, dist)``. Calculated using
            recurrence coefficients for stability.

    Examples:
        >>> Z = chaospy.Normal()
        >>> print(chaospy.around(chaospy.orth_ttr(4, Z), 4))
        [1.0, q0, q0^2-1.0, q0^3-3.0q0, q0^4-6.0q0^2+3.0]
    """
    polynomials, norms, _, _ = chaospy.quad.generate_stieltjes(
        dist=dist, order=numpy.max(order), retall=True, **kws)

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
