"""
Methods for generating orthogonal polynomial expansions.

Methods
-------
orth_select     Wrapper that selects between the methods
orth_gs         Gram-Schmidts procedure
orth_alpha      Based on the module orth_eval
orth_ttr        Three Terms Recursion (if dist support it)
orth_chol       Cholesky decomposition
orth_pcd        Pivot Cholesky decomposition
PCD             Pivot Cholesky decomposition (for matrices)
"""

import numpy as np

import chaospy as cp

__all__ = [
# "orth_select",
"orth_gs",
# "orth_hybrid",
"orth_ttr",
"orth_chol",
# "orth_pcd",
# "orth_svd",
"orth_bert",
"norm",
]

def orth_gs(order, dist, normed=False, sort="GR", **kws):
    """
Gram-Schmidt process for generating orthogonal
polynomials in a weighted function space.

Parameters
----------
order : int, Poly
    The upper polynomial order.
    Alternative a custom polynomial basis can be used.
dist : Dist
    Weighting distribution(s) defining orthogonality.
normed : bool
    If True orthonormal polynomials will be used instead of monic.
sort : str
    Ordering argument passed to poly.basis.
    If custom basis is used, argument is ignored.
kws : optional
    Keyword argument passed to dist.mom if the moments need to be
    estimated.

Returns
-------
P : Poly
    The orthogonal polynomial expansion.

Examples
--------
>>> Z = cp.J(cp.Normal(), cp.Normal())
>>> print(cp.orth_gs(2, Z))
[1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]
    """

    dim = len(dist)

    if isinstance(order, int):
        if order==0:
            return cp.poly.Poly(1, dim=dim)
        basis = cp.poly.basis(0, order, dim, sort)
    else:
        basis = order

    basis = list(basis)

    P = [basis[0]]

    if normed:
        for i in range(1,len(basis)):

            for j in range(i):
                tmp = P[j]*cp.descriptives.E(basis[i]*P[j], dist, **kws)
                basis[i] = basis[i] - tmp

            g = cp.descriptives.E(P[-1]**2, dist, **kws)
            if g<=0:
                print("Warning: Polynomial cutoff at term %d" % i)
                break
            basis[i] = basis[i]/np.sqrt(g)
            P.append(basis[i])

    else:

        G = [1.]
        for i in range(1,len(basis)):
            for j in range(i):
                tmp = P[j]*(cp.descriptives.E(basis[i]*P[j], dist, **kws) / G[j])
                basis[i] = basis[i] - tmp

            G.append(cp.descriptives.E(P[-1]**2, dist, **kws))
            if G[-1]<=0:
                print("Warning: Polynomial cutoff at term %d" % i)
                break
            P.append(basis[i])

    return cp.poly.Poly(P, dim=dim, shape=(len(P),))


def orth_ttr(order, dist, normed=False, sort="GR", retall=False, **kws):
    """
Create orthogonal polynomial expansion from three terms recursion
formula.

Parameters
----------
order : int
    Order of polynomial expansion
dist : Dist
    Distribution space where polynomials are orthogonal
    If dist.ttr exists, it will be used,
    othervice Clenshaw-Curtis integration will be used.
    Must be stochastically independent.
normed : bool
    If True orthonormal polynomials will be used instead of monic.
sort : str
    Polynomial sorting. Same as in basis
retall : bool
    If true return norms as well.
kws : optional
    Keyword argument passed to stieltjes method.

Returns
-------
orth[, norms]

orth : Poly
    Orthogonal polynomial expansion
norms : np.ndarray
    Norms of the orthogonal expansion on the form
    E(orth**2, dist)
    Calculated using recurrence coefficients for stability.

Examples
--------
>>> Z = cp.Normal()
>>> print(cp.orth_ttr(4, Z))
[1.0, q0, q0^2-1.0, q0^3-3.0q0, q0^4-6.0q0^2+3.0]
    """

    P, norms, A, B = cp.quadrature.stieltjes(
        dist, order, retall=True, **kws)

    if normed:
        for i in range(len(P)):
            P[i] = P[i]/np.sqrt(norms[:,i])
        norms = norms**0

    dim = len(dist)
    if dim > 1:
        Q, G = [], []
        indices = cp.bertran.bindex(0,order,dim,sort)
        for I in indices:
            q = P[I[0]][0]
            for i in range(1, dim):
                q = q * P[I[i]][i]
            Q.append(q)

        if retall:
            for I in indices:
                g = [norms[i,I[i]] for i in range(dim)]
                G.append(np.prod(g))
        P = Q

    else:
        G = norms[0]

    P = cp.poly.flatten(cp.poly.Poly(P))

    if retall:
        return P, np.array(G)
    return P


def orth_chol(order, dist, normed=True, sort="GR", **kws):
    """
Create orthogonal polynomial expansion from Cholesky decomposition

Parameters
----------
order : int
    Order of polynomial expansion
dist : Dist
    Distribution space where polynomials are orthogonal
normed : bool
    If True orthonormal polynomials will be used instead of monic.
sort : str
    Ordering argument passed to poly.basis.
    If custom basis is used, argument is ignored.
kws : optional
    Keyword argument passed to dist.mom.

Examples
--------
>>> Z = cp.Normal()
>>> print(cp.around(cp.orth_chol(3, Z), 4))
[1.0, q0, 0.7071q0^2-0.7071, 0.4082q0^3-1.2247q0]
    """

    dim = len(dist)
    basis = cp.poly.basis(1,order,dim, sort)
    C = cp.descriptives.Cov(basis, dist)
    N = len(basis)

    L, e = cp.cholesky.gill_king(C)
    Li = np.linalg.inv(L.T).T
    if not normed:
        Li /= np.repeat(np.diag(Li), len(Li)).reshape(Li.shape)
    E_ = -np.sum(Li*cp.descriptives.E(basis, dist, **kws), -1)
    coefs = np.empty((N+1, N+1))
    coefs[1:,1:] = Li
    coefs[0,0] = 1
    coefs[0,1:] = 0
    coefs[1:,0] = E_
    coefs = coefs.T

    out = {}
    out[(0,)*dim] = coefs[0]
    for i in range(N):
        I = basis[i].keys[0]
        out[I] = coefs[i+1]

    P = cp.poly.Poly(out, dim, coefs.shape[1:], float)

    return P



def orth_bert(N, dist, normed=False, sort="GR"):
    """
# Stabilized process for generating orthogonal
polynomials in a weighted function space.
Add a comment to this line

Parameters
----------
N : int
    The upper polynomial order.
dist : Dist
    Weighting distribution(s) defining orthogonality.
    normed

Returns
-------
P : Poly
    The orthogonal polynomial expansion.

Examples
--------
# >>> Z = cp.MvNormal([0,0], [[1,.5],[.5,1]])
# >>> P = orth_bert(2, Z)
# >>> print(P)
# [1.0, q0, -0.5q0+q1, q0^2-1.0, -0.5q0^2+q0q1, 0.25q0^2+q1^2-q0q1-0.75]
    """
    dim = len(dist)
    sort = sort.upper()

    # Start orthogonalization
    x = cp.poly.basis(1,1,dim)
    if not ("R" in sort):
        x = x[::-1]
    foo = cp.bertran.fourier.FourierRecursive(dist)

    # Create order=0
    pool = [cp.poly.Poly(1, dim=dim, shape=())]

    # start loop
    M = cp.bertran.terms(N,dim)
    for i in range(1, M):

        par, ax0 = cp.bertran.parent(i, dim)
        gpar, ax1 = cp.bertran.parent(par, dim)
        oneup = cp.bertran.child(0, dim, ax0)

        # calculate rank to cut some terms
        rank = cp.bertran.multi_index(i, dim)
        while rank[-1]==0: rank = rank[:-1]
        rank = dim - len(rank)

        candi = x[ax0]*pool[par]

        for j in range(gpar, i):

            # cut irrelevant term
            if rank and np.any(cp.bertran.multi_index(j, dim)[-rank:]):
                continue

            A = foo(oneup, par, j)
            P = pool[j]

            candi = candi - P*A

        if normed:
            candi = candi/np.sqrt(foo(i, i, 0))

        pool.append(candi)

    if "I" in sort:
        pool = pool[::-1]

    P = cp.poly.Poly([_.A for _ in pool], dim, (cp.bertran.terms(N, dim),))
    return P


## DEPRICATED AFTER THIS MARKER ##

def orth_pcd(order, dist, eps=1.e-16, normed=False, **kws):
    """
Create orthogonal polynomial expansion from pivoted Cholesky
decompostion.

Parameters
----------
order : int
    Order of polynomial expansion
dist : Dist
    Distribution space where polynomials are orthogonal
normed : bool
    If True orthonormal polynomials will be used instead of monic.
**kws : optional
    Extra keywords passed to dist.mom

Examples
--------
#  >>> Z = cp.Normal()
#  >>> print(cp.orth_pcd(2, Z))
#  [1.0, q0^2-1.0, q0]
    """
    raise DeprecationWarning("Obsolete. Use orth_chol instead.")

    dim = len(dist)
    basis = cp.poly.basis(1,order,dim)
    C = cp.descriptives.Cov(basis, dist)
    N = len(basis)

    L, P = pcd(C, approx=1, pivot=1, tol=eps)
    Li = np.dot(P, np.linalg.inv(L.T))

    if normed:
        for i in range(N):
            Li[:,i] /= np.sum(Li[:,i]*P[:,i])
    E_ = -cp.poly.sum(cp.descriptives.E(basis, dist, **kws)*Li.T, -1)

    coefs = np.zeros((N+1, N+1))
    coefs[1:,1:] = Li
    coefs[0,0] = 1
    coefs[0,1:] = E_

    out = {}
    out[(0,)*dim] = coefs[0]
    basis = list(basis)
    for i in range(N):
        I = basis[i].keys[0]
        out[I] = coefs[i+1]

    P = cp.poly.Poly(out, dim, coefs.shape[1:], float)
    return P


def orth_svd(order, dist, eps=1.e-300, normed=False, **kws):
    """
Create orthogonal polynomial expansion from pivoted Cholesky
decompostion. If eigenvalue of covariance matrix is bellow eps, the
polynomial is subset.

Parameters
----------
order : int
    Order of polynomial expansion
dist : Dist
    Distribution space where polynomials are orthogonal
eps : float
    Threshold for when to subset the expansion.
normed : bool
    If True, polynomial will be orthonormal.
**kws : optional
    Extra keywords passed to dist.mom

Examples
--------
#  >>> Z = cp.Normal()
#  >>> print(cp.orth_svd(2, Z))
#  [1.0, q0^2-1.0, q0]
    """
    raise DeprecationWarning("Obsolete")

    dim = len(dist)
    if isinstance(order, cp.poly.Poly):
        basis = order
    else:
        basis = cp.poly.basis(1,order,dim)

    basis = list(basis)
    C = cp.descriptives.Cov(basis, dist, **kws)
    L, P = pcd(C, approx=0, pivot=1, tol=eps)
    N = L.shape[-1]

    if len(L)!=N:
        I = [_.tolist().index(1) for _ in P]
        b_ = [0]*N
        for i in range(N):
            b_[i] = basis[I[i]]
        basis = b_
        C = cp.descriptives.Cov(basis, dist, **kws)
        L, P = pcd(C, approx=0, pivot=1, tol=eps)
        N = L.shape[-1]

    basis = cp.poly.Poly(basis)

    Li = cp.utils.rlstsq(L, P, alpha=1.e-300).T
    E_ = -cp.poly.sum(cp.descriptives.E(basis, dist, **kws)*Li.T, -1)

    coefs = np.zeros((N+1, N+1))
    coefs[1:,1:] = Li
    coefs[0,0] = 1
    coefs[0,1:] = E_

    out = {}
    out[(0,)*dim] = coefs[0]
    for i in range(N):
        I = basis[i].keys[0]
        out[I] = coefs[i+1]

    P = cp.poly.Poly(out, dim, coefs.shape[1:], float)

    if normed:
        norm = np.sqrt(cp.descriptives.Var(P, dist, **kws))
        norm[0] = 1
        P = P/norm

    return P

def orth_hybrid(order, dist, eps=1.e-30, normed=True, **kws):
    """
Create orthogonal polynomial expansion from Cholesky decompostion

Parameters
----------
order : int
    Order of polynomial expansion
dist : Dist
    Distribution space where polynomials are orthogonal
eps : float
    The accuracy if PCD is used
normed : bool
    If True orthonormal polynomials will be used instead of monic.
kws : optional
    Keyword argument passed to stieltjes.

Examples
--------
#  >>> Z = cp.Normal()
#  >>> print(cp.orth_chol(3, Z))
#  [1.0, q0, 0.707106781187q0^2-0.707106781187, 0.408248290464q0^3-1.22474487139q0]
    """
    raise DeprecationWarning("Obsolete. Use orth_chol instead.")

    if order==1:
        return orth_svd(order, dist, eps, normed, **kws)

    dim = len(dist)
    basis = cp.poly.basis(1,order,dim)

    C = cp.descriptives.Cov(basis, dist)

    L, P = pcd(C, approx=0, pivot=1, tol=eps)
    eig = np.array(np.sum(np.cumsum(P, 0), 0)-1, dtype=int)

    for i in range(len(C)-1):
        try:
            I,J = np.meshgrid(eig[i:], eig[i:])
            D = C[J,I]
            L = np.linalg.cholesky(D)
            break
        except np.linalg.LinAlgError:
            continue

    if i==(len(C)-2):
        return orth_svd(order, dist, eps, normed, **kws)
    if i:
        print("subset", i)

    basis = basis[eig[i:]]
    N = len(basis)

    Li = np.linalg.inv(L.T).T
    Ln = Li/np.repeat(np.diag(Li), len(Li)).reshape(Li.shape)
    E_ = -np.sum(Ln*cp.descriptives.E(basis, dist, **kws), -1)
    coefs = np.empty((N+1, N+1))
    coefs[1:,1:] = Ln
    coefs[0,0] = 1
    coefs[0,1:] = 0
    coefs[1:,0] = E_
    coefs = coefs.T

    out = {}
    out[(0,)*dim] = coefs[0]
    for i in range(N):
        I = basis[i].keys[0]
        out[I] = coefs[i+1]

    P = cp.poly.Poly(out, dim, coefs.shape[1:], float)

    if normed:
        norm = np.sqrt(cp.descriptives.Var(P, dist, **kws))
        norm[0] = 1
        P = P/norm

    return P

def norm(order, dist, orth=None):

    dim = len(dist)
    try:
        if dim>1:
            norms = np.array([norm(order+1, D) for D in dist])
            Is = cp.bertran.bindex(order, dim)
            out = np.ones(len(Is))

            for i in range(len(Is)):
                I = Is[i]
                for j in range(dim):
                    if I[j]:
                        out[i] *= norms[j, I[j]]
            return out

        K = range(1,order+1)
        ttr = [1.] + [dist.ttr(k)[1] for k in K]
        return np.cumprod(ttr)

    except NotImplementedError:

        if orth is None:
            orth = orth_chol(order, dist)
        return cp.descriptives.E(orth**2, dist)

def orth_select(orth):
    """
Select method for orthogonalization.

Parameters
----------
orth : int, str
    Legal values are:

    no      key         method
    --      ---         ------
     1      c, chol     Cholesky decompostion
     2      g, gs       Gram-Schmidt orthogonalization
     3      p, pcd      Pivoted Cholesky decompostion
     4      t, ttr      Three terms recursion (requires
                        independent vars)

Returns
-------
orth_func : callable(order, dist, *args, **kws)
    Each method has its own documentation.
    order : int
        Order of the expansion
    dist : Dist
        Distribution
    args, kws : optinal
        Method specific values.
    """
#      print("orth_select depricated")

    if isinstance(orth, str):
        orth = orth.lower()

    if orth in (1, "c", "chol"):
        return orth_chol
    if orth in (2, "g", "gs"):
        return orth_gs
    if orth in (3, "p", "pcd"):
        return orth_pcd
    if orth in (4, "t", "ttr"):
        return orth_ttr
    if orth in (5, "s", "svd"):
        return orth_svd


def lagrange_polynomial(X, sort="GR"):
    """
Lagrange Polynomials

X : array_like
    Sample points where the lagrange polynomials shall be 
    """

    X = np.asfarray(X)
    if len(X.shape)==1:
        X = X.reshape(1,X.size)
    dim,size = X.shape

    order = 1
    while cp.bertran.terms(order, dim)<=size: order += 1

    indices = np.array(cp.bertran.bindex(1, order, dim, sort)[:size])
    s,t = np.mgrid[:size, :size]

    M = np.prod(X.T[s]**indices[t], -1)
    det = np.linalg.det(M)
    if det==0:
        raise np.linalg.LinAlgError("invertable matrix")

    v = cp.poly.basis(1, order, dim, sort)[:size]

    coeffs = np.zeros((size, size))

    if size==2:
        coeffs = np.linalg.inv(M)

    else:
        for i in range(size):
            for j in range(size):
                coeffs[i,j] += np.linalg.det(M[1:,1:])
                M = np.roll(M, -1, axis=0)
            M = np.roll(M, -1, axis=1)
        coeffs /= det

    return cp.poly.sum(v*(coeffs.T), 1)


if __name__=="__main__":
    import doctest
    import __init__ as cp
    doctest.testmod()
