import numpy

from .three_terms_recursion import orth_ttr
from .cholesky import orth_chol
from .gram_schmidt import orth_gs
from .lagrange import lagrange_polynomial

EXPANSION_NAMES = {
    "ttr": "three_terms_recursion", "three_terms_recursion": "three_terms_recursion",
    "chol": "cholesky", "cholesky": "cholesky",
    "gs": "gram_schmidt", "gram_schmidt": "gram_schmidt",
}
EXPANSION_FUNCTIONS = {
    "three_terms_recursion": orth_ttr,
    "cholesky": orth_chol,
    "gram_schmidt": orth_gs,
}


def generate_expansion(
    order,
    dist,
    rule="three_terms_recursion",
    normed=False,
    sort="G",
    cross_truncation=1.,
    **kws
):
    """
    Create orthogonal polynomial expansion.

    This function is a frontend wrapper for the three methods for creating
    orthogonal polynomials:

    ---------------------   ---------------------------------------------------
    Method                  Description
    =====================   ===================================================
    three_terms_recursion   Three terms recurrence coefficients gnerated using
                            Stieltjes and Golub-Welsch method. The most stable
                            of the methods, but do not work on dependent
                            distributions.
    ---------------------   ---------------------------------------------------
    gram_schmidt            Gram-Schmidt orthogonalization method applied on
                            polynomial expansions. Know for being numerically
                            unstable.
    ---------------------   ---------------------------------------------------
    cholesky                Orthogonalization through decorrelation of
                            the covariance matrix. Uses Gill-King's Cholesky
                            decomposition method for higher numerical
                            stability. Still not scalable to high number of
                            dimensions.
    ---------------------   ---------------------------------------------------

    Args:
        order (int):
            Order of polynomial expansion.
        dist (Dist):
            Distribution space where polynomials are orthogonal. If the method
            ``dist._ttr`` exists, it will be used.
        rule (str):
            The orthogonalization method used.
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
            Orthogonal polynomial expansion. norms of the orthogonal
            expansion on the form ``E(orth**2, dist)``. Calculated using
            recurrence coefficients for stability.

    Examples:
        >>> distribution = chaospy.Normal()
        >>> expansion, norms = generate_expansion(3, distribution, retall=True)
        >>> expansion
        polynomial([1.0, q0, -1.0+q0**2, -3.0*q0+q0**3])
        >>> norms
        array([1., 1., 2., 6.])
    """
    name = EXPANSION_NAMES[rule.lower()]
    expansion_function = EXPANSION_FUNCTIONS[name]
    return expansion_function(order, dist=dist, normed=normed, sort=sort,
                              cross_truncation=cross_truncation, **kws)
