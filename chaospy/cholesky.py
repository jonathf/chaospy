"""
Cholesky decomposition rutines.

Notes
=====

Compiled together by:
M. M. Forbes <michael.forbes@gmail.com>
J. Feinberg <jonathan@feinberg.no>

"""
from __future__ import division

from scipy import sparse
import numpy as np
from numpy.linalg import cholesky as chol_np

EPS = np.finfo(float).eps

__all__ = [
    "gill_murray_wright",
    "gill_king",
    "scnabel_eskow",
    "bastos_ohagen",
]


def gill_murray_wright(mat, eps=EPS):
    """
    Gill-Murray-Wright algorithm for pivoting modified Cholesky decomposition.

    Return `(perm, lowtri, error)` such that
    `perm.T*mat*perm = lowtri*lowtri.T - diag(error)`.

    Parameters
    ----------
    mat : array
        Must be a non-singular and symmetric matrix
    eps : float
        Error tolerance used in algorithm.

    Returns
    -------
    perm : 2d array
        Permutation matrix used for pivoting.
    lowtri : 2d array
        Lower triangular factor
    error : 1d array
        Positive diagonals of shift matrix `error`.

    Examples
    --------
    >>> mat = [[4, 2, 1], [2, 6, 3], [1, 3, -.004]]
    >>> perm, lowtri, error = gill_murray_wright(mat)
    >>> perm, lowtri = np.matrix(perm), np.matrix(lowtri)
    >>> print(error)
    [ 0.     0.     3.008]
    >>> print(np.allclose(perm.T*mat*perm, lowtri*lowtri.T-np.diag(error)))
    True

    Notes
    -----
    The Gill, Murray, and Wright modified Cholesky algorithm.

    Algorithm 6.5 from page 148 of 'Numerical Optimization' by Jorge
    Nocedal and Stephen J. Wright, 1999, 2nd ed.
    """
    mat = np.asfarray(mat)
    size = mat.shape[0]

    # Calculate gamma(mat) and xi(mat).
    gamma = 0.0
    xi = 0.0
    for i in range(size):
        gamma = max(abs(mat[i, i]), gamma)
        for j in range(i+1, size):
            xi = max(abs(mat[i, j]), xi)

    # Calculate delta and beta.
    delta = eps * max(gamma + xi, 1.0)
    if size == 1:
        beta = np.sqrt(max(gamma, eps))
    else:
        beta = np.sqrt(max(gamma, xi / np.sqrt(size*size - 1.0), eps))

    # Initialise data structures.
    mat_a = 1.0 * mat
    mat_r = 0.0 * mat
    error = np.zeros(size, dtype=float)
    perm = np.eye(size, dtype=float)

    # Main loop.
    for j in range(size):

        # Row and column swapping, find the index > j of the largest
        # diagonal element.
        dia = j
        for i in range(j+1, size):
            if abs(mat_a[i, i]) >= abs(mat_a[dia, dia]):
                dia = i

        # Interchange row and column j and dia (if j != dia).
        if dia != j:
            # Temporary permutation matrix for swaping 2 rows or columns.
            perm_new = np.eye(size, dtype=float)

            # Modify the permutation matrix perm by swaping columns.
            perm_row = 1.0*perm[:, dia]
            perm[:, dia] = perm[:, j]
            perm[:, j] = perm_row

            # Modify the permutation matrix p by swaping rows (same as
            # columns because p = pT).
            row_p = 1.0 * perm_new[dia]
            perm_new[dia] = perm_new[j]
            perm_new[j] = row_p

            # Permute mat_a and r (p = pT).
            mat_a = np.dot(perm_new, np.dot(mat_a, perm_new))
            mat_r = np.dot(mat_r, perm_new)

        # Calculate a_pred.
        theta_j = 0.0
        if j < size-1:
            for i in range(j+1, size):
                theta_j = max(theta_j, abs(mat_a[j, i]))
        a_pred = max(abs(mat_a[j, j]), (theta_j/beta)**2, delta)

        # Calculate error (not really needed!).
        error[j] = a_pred - mat_a[j, j]

        # Calculate row j of r and update a.
        mat_r[j, j] = np.sqrt(a_pred) # Damned sqrt introduces roundoff error.
        for i in range(j+1, size):
            mat_r[j, i] = mat_a[j, i] / mat_r[j, j]
            for k in range(j+1, i+1):

                # Keep matrix a symmetric:
                mat_a[i, k] = mat_a[k, i] = mat_a[k, i] - mat_r[j, i] * mat_r[j, k]

    # The Cholesky factor of mat.
    return perm, mat_r.T, error



def gill_king(A, eps=EPS):
    """
    Gill-King algorithm for modified cholesky decomposition.

    Parameters
    ----------
    A : array
        Must be a non-singular and symmetric matrix
        If sparse, the result will also be sparse.
    eps : float
        Error tolerance used in algorithm.


    Return `(L, e)` such that `M = A + diag(e) = dot(L, L.T)` where

    Returns
    -------
    L : 2d array
        Lower triangular Cholesky factor.
    e : 1d array
        Diagonals of correction matrix `E`.

    Examples
    --------
    >>> A = [[4, 2, 1], [2, 6, 3], [1, 3, -.004]]
    >>> L, e = gill_king(A)
    >>> print(np.allclose(A+np.diag(e), np.dot(L, L.T)))
    True
    >>> print(e)
    [  0.00000000e+00   8.88178420e-16   3.00800000e+00]

    Notes
    -----
    Algorithm 3.4 of 'Numerical Optimization' by Jorge Nocedal and
    Stephen J. Wright

    This is based on the MATLAB code from Michael
    L. Overton <overton@cs.nyu.edu>:
    http://cs.nyu.edu/overton/g22_opt/codes/cholmod.m
    Modified last 2005
    """

    if not sparse.issparse(A):
        A = np.asfarray(A)
    assert np.allclose(A, A.T)
    n = A.shape[0]
    A_diag = A.diagonal()
    gamma = abs(A_diag).max()
    xi = abs(A - np.diag(A_diag)).max() # max offidagonal entry

    delta = eps*max(gamma + xi, 1)
    beta = np.sqrt(max(gamma, xi/n, eps))

    # initialize d and L

    d = np.zeros(n, dtype=float)
    if sparse.issparse(A):
        L = sparse.eye(*A.shape)
    else:
        L = np.eye(n);

    # there are no inner for loops, everything implemented with
    # vector operations for a reasonable level of efficiency
    for j in range(n):
        if j == 0:
            K = []     # column index: all columns to left of diagonal
                       # d(K) doesn't work in case K is empty
        else:
            K = np.s_[:j]

        djtemp = A[j, j] - np.dot(L[j, K], d[K]*L[j, K].T) # C(j, j) in book
        if j < n - 1:
            I = np.s_[j+1:n]  # row index: all rows below diagonal
            Ccol = A[I, j] - np.dot(L[I, K], d[K]*L[j, K].T) # C(I, j) in book
            theta = abs(Ccol).max()
            # guarantees d(j) not too small and L(I, j) not too big
            # in sufficiently positive definite case, d(j) = djtemp
            d[j] = max(abs(djtemp), (theta/beta)**2, delta)
            L[I, j] = Ccol/d[j]
        else:
            d[j] = max(abs(djtemp), delta)

    # convert to usual output format: replace L by L*sqrt(D) and transpose
    for j in range(n):
        L[:, j] = L[:, j]*np.sqrt(d[j]) # L = L*diag(sqrt(d)) bad in sparse case

    e = (np.dot(L, L.T) - A).diagonal()

    return L, e


def schnabel_eskow(A, eps=EPS):
    """
    Scnabel-Eskow algorithm for modified Cholesky factorisation algorithm.

    Parameters
    ----------
    A : array
        Must be a non-singular and symmetric matrix
        If sparse, the result will also be sparse.
    eps : float
        Error tolerance used in algorithm.

    Return `(P, L, e)` such that `P.T*A*P = L*L.T - diag(e)`.

    Returns
    -------
    P : 2d array
        Permutation matrix used for pivoting.
    L : 2d array
        Lower triangular factor
    e : 1d array
        Positive diagonals of shift matrix `e`.

    Examples
    --------
    >>> A = [[4, 2, 1], [2, 6, 3], [1, 3, -.004]]
    >>> P, L, e = schnabel_eskow(A)
    >>> P, L = np.matrix(P), np.matrix(L)
    >>> print(e)
    [ 0.          1.50402929  1.50402929]
    >>> print(np.allclose(P.T*A*P, L*L.T-np.diag(e)))
    True

    Notes
    -----
    Schnabel, R. B. and Eskow, E. 1999, A revised modifed cholesky
    factorisation algorithm. SIAM J. Optim. 9, 1135-1148.
    """
    A = np.asfarray(A)
    tau = eps**(1/3)
    tau_bar = tau*tau
    mu = 0.1

    # Create the matrix e and A.
    n = A.shape[0]
    A0 = A
    A = 1*A
    e = np.zeros(n, dtype=float)

    # Permutation matrix.
    P = np.eye(n)

    # Calculate gamma.
    gamma = abs(A.diagonal()).max()

    # Phase one, A potentially positive definite.
    #############################################

    def invariant(A, k):
        """Return `True` if the invariant is satisfied."""
        A_ = np.eye(n)
        L_ = np.eye(n)
        A_[k:, k:] = np.triu(A[k:, k:], 0) + np.triu(A[k:, k:], 1).T
        L_[:, :k] = np.tril(A[:, :k])
        return np.allclose(np.dot(L_, np.dot(A_, L_.T)),
                           np.dot(P, np.dot(A0, P.T)) + np.diag(e))


    def jiter_factor(A, j, P, e):
        """Perform jth iteration of factorisation.
        """
        assert invariant(A, j)

        A[j, j] = np.sqrt(A[j, j])
        A[j+1:, j] /= A[j, j]
        A[j+1:, j+1:] -= A[j+1:, j:j+1]*A[j+1:, j:j+1].T
        A[j, j+1:] = 0

        assert invariant(A, j+1)

    def permute(A, P, i, j):
        """Exchange rows and columns i and j of A and recored the
        permutation in P"""
        p = np.arange(n, dtype=int)
        if i != j:
            p[[i, j]] = j, i

            P[::] = P[p, :]
            A[::] = A[p, :]
            A[::] = A[:, p]

    def exec_phasetwo(A, P, e, j):
        """Phase 2 of the algorithm, A not positive definite."""
        if j == n:
            # delta = e[n].
            delta = -A[n-1, n-1] + max(-tau*A[n-1, n-1]/(1 - tau),
                                       tau_bar*gamma)

            e[n-1] = delta
            A[n-1, n-1] += delta
            A[n-1, n-1] = np.sqrt(A[n-1, n-1])

        else:
            # Number of iterations performed in phase one (less 1).
            k = j - 1

            # Calculate the lower Gerschgorin bounds of Ak+1.
            tmp = A[k+1:, k+1:]
            g = tmp.diagonal().copy()
            tmp = abs(np.tril(tmp, -1))
            g -= tmp.sum(axis=0)
            g -= tmp.sum(axis=1)

            # Modified Cholesky decomposition.
            delta_prev = 0.0
            for j in range(k+1, n-2):
                # Pivot on the maximum lower Gerschgorin bound
                # estimate.
                i = j + np.argmax(g[j-(k+1):])

                # Interchange row and column i and j.
                permute(A, P, i, j)

                # Calculate e[j] and add to diagonal.
                normj = abs(A[j+1:, j]).sum()
                delta = max(0.0,
                            -A[j, j] + max(normj, tau_bar*gamma),
                            delta_prev)  # delta = E[n].

                if delta > 0.0:
                    A[j, j] += delta
                    delta_prev = delta
                e[j] = delta

                # Update Gerschgorin bound estimates.
                if A[j, j] != normj:
                    temp = 1.0 - normj / A[j, j]
                    g[j-k:] += abs(A[j+1:, j])*temp

                # Perform jth iteration of factorisation.
                jiter_factor(A, j, P, e)

            # Final 2*2 submatrix.
            mini = A[-2:, -2:]
            mini[1, 0] = mini[0, 1]
            eigs = np.sort(np.linalg.eigvalsh(mini))
            delta = max(0,
                        -eigs[0] + max(tau*(eigs[1] - eigs[0])/(1 - tau),
                                       tau_bar*gamma),
                        delta_prev)
            if delta > 0.0:
                A[n-2, n-2] += delta
                A[n-1, n-1] += delta
                delta_prev = delta
            e[n-2] = e[n-1] = delta

            A[n-2, n-2] = np.sqrt(A[n-2, n-2])
            A[n-1, n-2] /= A[n-2, n-2]
            A[n-1, n-1] = np.sqrt(A[n-1, n-1] - A[n-1, n-2]**2)

    for j in range(n):

        # Calculate max_Aii and min_Aii
        _d = A.diagonal()[j:]
        max_Aii = _d.max()
        min_Aii = _d.min()

        # Test for phase 2, A not positive definite.
        if max_Aii < tau_bar * gamma or min_Aii < - mu * max_Aii:
            exec_phasetwo(A, P, e, j)
            break
        else:
            # Pivot on maximum diagonal of remaining submatrix.
            i = j + np.argmax(A.diagonal()[j:])

            # Interchange row and column i and j.
            permute(A, P, i, j)

            # Test for phase 2 again.
            min_num = 1e99
            A_diag = A.diagonal()
            if j + 1 < n:
                min_num = (A_diag[i] -
                           A[i, j+1:]**2/A_diag[j+1:]).min()
            else:
                min_num = A_diag[i]

            if j+1 <= n and min_num < - mu * gamma:
                exec_phasetwo(A, P, e, j)
                break

            # Perform jth iteration of factorisation.
            else:
                jiter_factor(A, j, P, e)

    # The Cholesky factor of A.
    return P, np.tril(A), e


def bastos_ohagen(A, eps=EPS):
    """
    Bastos-O'Hagen algorithm for modified Cholesky decomposition.

    Parameters
    ----------
    A : array_like
        Input matrix.
    eps : float
        Tollerence value for the eigen values. Values smaller than
        tol*np.diag(A).max() are considered to be zero.

    Return `(P, L, E)` such that `P.T*A*P = L*L.T - E`.

    Returns
    -------
    P : np.ndarray
        Permutation matrix
    R : np.ndarray
        Upper triangular decompostion
    E : np.ndarray
        Error matrix

    Examples
    --------
    >>> A = [[4, 2, 1], [2, 6, 3], [1, 3, -.004]]
    >>> P, L, E = bastos_ohagen(A)
    >>> P, L = np.matrix(P), np.matrix(L)
    >>> print(np.allclose(P.T*A*P, L*L.T-E))
    True

    Notes
    -----
    Algoritum 3 from ``Pivoting Cholesky Decomposition applied to
    Emulation and Validation of computer models'' by L.S. Bastos and
    and A. O'Hagan, 2007
    """
    A = np.asfarray(A).copy()
    B = np.asfarray(A).copy()
    a0 = np.diag(A).max()
    assert len(A.shape)==2
    n = len(A)

    R = np.zeros((n, n))
    piv = np.arange(n)

    for k in range(n):

        q = np.argmax( np.diag(A[k:, k:]) ) + k

        if A[q, q] <= np.abs(a0*eps):

            if not k:
                raise ValueError("Purly negative definite")

            for j in range(k, n):
                R[j, j] = R[j-1, j-1]/float(j)

            break

        if True:
            tmp = A[:, k].copy(); A[:, k] = A[:, q]; A[:, q] = tmp
            tmp = R[:, k].copy(); R[:, k] = R[:, q]; R[:, q] = tmp
            tmp = A[k, :].copy(); A[k, :] = A[q, :]; A[q, :] = tmp
            piv[k], piv[q] = piv[q], piv[k]

        R[k, k] = np.sqrt(A[k, k])
        r = A[k, k+1:]/R[k, k]
        R[k, k+1:] = r
        A[k+1:, k+1:] -= np.outer(r, r)

    P = np.zeros((n, n))
    for k in range(n):
        P[k, piv[k]] = 1.

    E = np.dot(R.T, R) - np.dot(np.dot(P.T, B), P)

    return P, R.T, E


if __name__=="__main__":
    import doctest
    doctest.testmod()
