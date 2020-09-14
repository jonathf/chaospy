"""Frontend function for generating polynomial expansions."""
from .three_terms_recurrence import orth_ttr
from .cholesky import orth_chol
from .gram_schmidt import orth_gs
from .lagrange import lagrange_polynomial

EXPANSION_NAMES = {
    "ttr": "three_terms_recurrence", "three_terms_recurrence": "three_terms_recurrence",
    "chol": "cholesky", "cholesky": "cholesky",
    "gs": "gram_schmidt", "gram_schmidt": "gram_schmidt",
}
EXPANSION_FUNCTIONS = {
    "three_terms_recurrence": orth_ttr,
    "cholesky": orth_chol,
    "gram_schmidt": orth_gs,
}


def generate_expansion(
        order,
        dist,
        rule="three_terms_recurrence",
        normed=False,
        graded=True,
        reverse=True,
        cross_truncation=1.,
        sort=None,
        **kws
):
    """
    Create orthogonal polynomial expansion.

    This function is a frontend wrapper for the three methods for creating
    orthogonal polynomials:

    +------------------------+-------------------------------------------------+
    | Algorithm              | Description                                     |
    +------------------------+-------------------------------------------------+
    | three_terms_recurrence | Three terms recurrence coefficients generated   |
    |                        | using Stieltjes and Golub-Welsch method. The    |
    |                        | most stable of the methods, but do not work on  |
    |                        | dependent distributions.                        |
    +------------------------+-------------------------------------------------+
    | gram_schmidt           | Gram-Schmidt orthogonalization method applied   |
    |                        | on polynomial expansions. Know for being        |
    |                        | numerically unstable.                           |
    +------------------------+-------------------------------------------------+
    | cholesky               | Orthogonalization through decorrelation of the  |
    |                        | covariance matrix. Uses Gill-King's Cholesky    |
    |                        | decomposition method for higher numerical       |
    |                        | stability. Still not scalable to high number of |
    |                        | dimensions.                                     |
    +------------------------+-------------------------------------------------+

    Args:
        order (int):
            Order of polynomial expansion.
        dist (Distribution):
            Distribution space where polynomials are orthogonal. If the method
            ``dist._ttr`` exists, it will be used.
        rule (str):
            The orthogonalization method used.
        normed (bool):
            If True orthonormal polynomials will be used.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6, and will
            therefore be consider larger than both ``q0**2*q1*q2``,
            ``q0*q1**2*q2`` and ``q0*q1*q2**2``,
            which all have exponent sum of 5.
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
            Orthogonal polynomial expansion. norms of the orthogonal
            expansion on the form ``E(orth**2, dist)``. Calculated using
            recurrence coefficients for stability.

    Examples:
        >>> distribution = chaospy.Normal()
        >>> expansion, norms = generate_expansion(
        ...     3, distribution, retall=True)
        >>> expansion
        polynomial([1.0, q0, q0**2-1.0, q0**3-3.0*q0])
        >>> norms
        array([1., 1., 2., 6.])

    """
    name = EXPANSION_NAMES[rule.lower()]
    expansion_function = EXPANSION_FUNCTIONS[name]
    return expansion_function(order, dist=dist, normed=normed, graded=graded,
                              reverse=reverse, sort=sort,
                              cross_truncation=cross_truncation, **kws)
