"""
==============================
Cholesky decomposition rutines
==============================

Rutines
=======

Pivoted
-------

chol_bo         Bastos-O'Hagan, 2007
chol_gmw        Gill-Murray-Wright, 1981
chol_se         Scnabel-Eskow, 1999

Non-pivoted
-----------

chol_gk         Gill-King, 2004
chol_gkc        Gill-King, 2004, implemented in C
chol_gko        Gill-King, 2004, vectorized implemention w/support
                for sparse matrices by M. L. Overton
chol_np         Classical implementation from numpy.linalg library


Notes
=====

Compiled together by:
M. M. Forbes <michael.forbes@gmail.com>
J. Feinberg <jonathan@feinberg.no>

"""
from __future__ import division

import numpy as np
import scipy.weave
from scipy import sparse
from numpy.linalg import cholesky as chol_np

_FINFO = np.finfo(float)
_EPS = _FINFO.eps


__all__ = ['chol_gmw', 'chol_gk', 'chol_gkc', 'chol_gko',
    'chol_se', 'chol_bo', 'chol_np']


def chol_gmw(A, eps=_EPS):
    """
Pivoting Modified Cholesky decomposition using the
Gill-Murray-Wright algorithm

Return `(P, L, e)` such that `P.T*A*P = L*L.T - diag(e)`.

Parameters
----------
A : array
    Must be a non-singular and symmetric matrix
eps : float
    Error tolerance used in algorithm.

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
>>> P, L, e = chol_gmw(A)
>>> P, L = np.matrix(P), np.matrix(L)
>>> print(e)
[ 0.     0.     3.008]
>>> print(np.allclose(P.T*A*P, L*L.T-np.diag(e)))
True

Notes
-----
The Gill, Murray, and Wright modified Cholesky algorithm.

Algorithm 6.5 from page 148 of 'Numerical Optimization' by Jorge
Nocedal and Stephen J. Wright, 1999, 2nd ed.
    """
    A = np.asfarray(A)
    n = A.shape[0]

    # Calculate gamma(A) and xi(A).
    gamma = 0.0
    xi = 0.0
    for i in xrange(n):
        gamma = max(abs(A[i, i]), gamma)
        for j in xrange(i+1, n):
            xi = max(abs(A[i, j]), xi)

    # Calculate delta and beta.
    delta = eps * max(gamma + xi, 1.0)
    if n == 1:
        beta = np.sqrt(max(gamma, eps))
    else:
        beta = np.sqrt(max(gamma, xi / np.sqrt(n**2 - 1.0), eps))

    # Initialise data structures.
    a = 1.0 * A
    r = 0.0 * A
    e = np.zeros(n, dtype=float)
    P = np.eye(n, dtype=float)

    # Main loop.
    for j in xrange(n):

        # Row and column swapping, find the index > j of the largest
        # diagonal element.
        q = j
        for i in xrange(j+1, n):
            if abs(a[i, i]) >= abs(a[q, q]):
                q = i

        # Interchange row and column j and q (if j != q).
        if q != j:
            # Temporary permutation matrix for swaping 2 rows or columns.
            p = np.eye(n, dtype=float)

            # Modify the permutation matrix P by swaping columns.
            row_P = 1.0*P[:, q]
            P[:, q] = P[:, j]
            P[:, j] = row_P

            # Modify the permutation matrix p by swaping rows (same as
            # columns because p = pT).
            row_p = 1.0*p[q]
            p[q] = p[j]
            p[j] = row_p

            # Permute a and r (p = pT).
            a = np.dot(p, np.dot(a, p))
            r = np.dot(r, p)

        # Calculate dj.
        theta_j = 0.0
        if j < n-1:
            for i in xrange(j+1, n):
                theta_j = max(theta_j, abs(a[j, i]))
        dj = max(abs(a[j, j]), (theta_j/beta)**2, delta)

        # Calculate e (not really needed!).
        e[j] = dj - a[j, j]

        # Calculate row j of r and update a.
        r[j, j] = np.sqrt(dj)     # Damned sqrt introduces roundoff error.
        for i in xrange(j+1, n):
            r[j, i] = a[j, i] / r[j, j]
            for k in xrange(j+1, i+1):
                a[i, k] = a[k, i] = a[k, i] - r[j, i] * r[j, k]     # Keep matrix a symmetric.

    # The Cholesky factor of A.
    return P, r.T, e



def chol_gk(A, eps=_EPS):
    """
Modified Cholesky decomposition using the Gill-King rutine

Return `(L, e)` such that `M = A + diag(e) = dot(L, L.T)` where

1) `M` is safely symmetric positive definite (SPD) and well
   conditioned.
2) `e` is small (zero if `A` is already SPD and not much larger
   than the most negative eigenvalue of `A`)

.. math::
   \mat{A} + \diag{e} = \mat{L}\mat{L}^{T}

Parameters
----------
A : array
    Must be a non-singular and symmetric matrix
eps : float
    Error tolerance used in algorithm.

Returns
-------
L : 2d array
    Lower triangular Cholesky factor.
e : 1d array
    Diagonals of correction matrix `E`.


Examples
--------
>>> A = [[4, 2, 1], [2, 6, 3], [1, 3, -.004]]
>>> L, e = chol_gk(A)
>>> print(np.allclose(A+np.diag(e), np.dot(L, L.T)))
True
>>> print(e)
[ 0.     0.     3.008]

Notes
-----
``What to do When Your Hessian is Not Invertable: Alternatives to
Model Respecification in Nonlinear Estimation,''
J. Gill and G. King
Socialogical Methods and Research, Vol. 32, No. 1 (2004):54--87
    """

    # local i,j,k,n,sum,R,theta_j,norm_A,phi_j,delta,xi_j,gamm,E,beta_j;
    A = np.asfarray(A)
    n = A.shape[0]                        # n = rows(A);
    R = np.eye(n, dtype=float)             # R = eye(n);
    e = np.zeros(n, dtype=float)          # E = zeros(n, n);
    norm_A = abs(A).sum(axis=0).max()     # norm_A = maxc(sumc(abs(A)));
    gamm = abs(A.diagonal()).max()        # gamm = maxc(abs(diag(A))); 
    delta = eps*max(1, norm_A)           # delta = maxc(maxc(__macheps*norm_A~__macheps));
    for j in xrange(n):                   # for j (1, n, 1); 
        theta_j = 0                       #    theta_j = 0;
        for i in xrange(n):               #    for i (1, n, 1);
            sum_ = 0                      #       sum = 0;
            for k in xrange(i):           #       for k (1, (i-1), 1);	
                sum_ += R[k, i]*R[k, j]     #          sum = sum + R[k, i]*R[k, j];
                                          #       endfor;
            R[i, j] = (A[i, j]
                      - sum_)/R[i, i]      #       R[i, j] = (A[i, j] - sum)/R[i, i];
            theta_j = max(theta_j,        #       if (A[i, j] -sum) > theta_j;
                          A[i, j] - sum_)  #          theta_j = A[i, j] - sum;
                                          #       endif;
            if i > j:                     #       if i > j;
                R[i, j] = 0                #          R[i, j] = 0;
                                          #       endif;
                                          #    endfor;
        sum_ = 0                          #    sum = 0;
        for k in xrange(j):               #    for k (1, (j-1), 1);	
           sum_ += R[k, j]**2              #       sum = sum + R[k, j]^2;
                                          #    endfor;
        phi_j = A[j, j] - sum_             #    phi_j = A[j, j] - sum;
        if j+1 < n:                       #    if (j+1) <= n;
            xi_j = abs(A[j+1:, j]).max()   #       xi_j = maxc(abs(A[(j+1):n, j]));
        else:                             #    else;
            xi_j = abs(A[-1, j])           #       xi_j = maxc(abs(A[n, j]));
                                          #    endif;
        beta_j = np.sqrt(max(gamm,        #    beta_j = sqrt(maxc(maxc(gamm~(xi_j/n)~__macheps)));
                             xi_j/n,
                             eps))
        if delta >= max(abs(phi_j),       #    if delta >= maxc(abs(phi_j)~((theta_j^2)/(beta_j^2)));
                        (theta_j/beta_j)**2):
            e[j] = delta - phi_j          #       E[j, j] = delta - phi_j;
        elif abs(phi_j) >= max(delta,     #    elseif abs(phi_j) >= maxc(((delta^2)/(beta_j^2))~delta);
                               (delta/beta_j)**2):
            e[j] = abs(phi_j) - phi_j     #       E[j, j] = abs(phi_j) - phi_j;
        elif (max(delta, abs(phi_j)) <    #    elseif ((theta_j^2)/(beta_j^2)) >= maxc(delta~abs(phi_j));
              (theta_j/beta_j)**2):
            e[j] = ((theta_j/beta_j)**2
                    - phi_j)              #       E[j, j] = ((theta_j^2)/(beta_j^2)) - phi_j;
                                          #    endif;
        R[j, j] = np.sqrt(A[j, j]           #    R[j, j] = sqrt(A[j, j] - sum + E[j, j]);
                         - sum_ + e[j])   
                                          # endfor;
    return (R.T, e)                       # retp(R'R);


def chol_gkc(A, eps=_EPS):
    """
Modified Cholesky decomposition
using the Gill-King rutine written in C

Return `(L, e)` such that `M = A + diag(e) = dot(L, L.T)` where

1) `M` is safely symmetric positive definite (SPD) and well
   conditioned.
2) `e` is small (zero if `A` is already SPD and not much larger
   than the most negative eigenvalue of `A`)

.. math::
   \mat{A} + \diag{e} = \mat{L}\mat{L}^{T}

Parameters
----------
A : array
    Must be a non-singular and symmetric matrix
eps : float
    Error tolerance used in algorithm.

Returns
-------
L : 2d array
   Lower triangular Cholesky factor.
e : 1d array
   Diagonals of correction matrix `E`.

Examples
--------
>>> A = [[4, 2, 1], [2, 6, 3], [1, 3, -.004]]
>>> L, e = chol_gkc(A)
>>> print(np.allclose(A+np.diag(e), np.dot(L, L.T)))
True
>>> print(e)
[ 0.     0.     3.008]

Notes
-----
``What to do When Your Hessian is Not Invertable: Alternatives to
Model Respecification in Nonlinear Estimation,''
J. Gill and G. King
Socialogical Methods and Research, Vol. 32, No. 1 (2004):54--87
    """
    A = np.ascontiguousarray(A, dtype='d')
    n = A.shape[0]
    L = np.eye(n, dtype='d')
    e = np.zeros(n, dtype='d')
    norm_A = float(np.linalg.norm(A, np.inf))
    gamm = float(np.linalg.norm(A.diagonal(), np.inf))
    delta = float(eps*max(1, norm_A))
    code = r"""
int i, j, k;
double sum, tmp, theta_j2, phi_j, xi_j, beta_j2;

for (j=0;j<n;++j) {
  theta_j2 = 0;
  for (i=0;i<n;++i) {
    sum = 0;
    for (k=0;k<i;++k){
      sum += L2(i, k)*L2(j, k);
    }
    if (i <= j) {
      L2(j, i) = (A2(i, j) - sum)/L2(i, i);
    }
    theta_j2 = std::max(theta_j2, A2(i, j) - sum);
  }
  theta_j2 *= theta_j2;
  sum = 0;
  for (k=0;k<j;++k) {
    sum += L2(j, k)*L2(j, k);
  }

  phi_j = A2(j, j) - sum;
  if (j + 1 < n) {
    xi_j = 0;
    for (k=j+1;k<n;++k) {
       xi_j = std::max(tmp, fabs(A2(k, j)));
    }
  } else {
    xi_j = fabs(A2(n-1, j));
  }

  beta_j2 = xi_j/n;
  beta_j2 = (beta_j2 < gamm)?gamm:beta_j2;
  beta_j2 = (beta_j2 < %(eps)d)?%(eps)d:beta_j2;

  if (delta >= std::max(fabs(phi_j), theta_j2/beta_j2)) {
    e[j] = delta - phi_j;
  } else if (fabs(phi_j) >= std::max(delta, delta*delta/beta_j2)) {
    e[j] = fabs(phi_j) - phi_j;
  } else if (std::max(delta, fabs(phi_j)) < theta_j2/beta_j2) {
    e[j] = theta_j2/beta_j2 - phi_j;
  }
  L2(j, j) = sqrt(A2(j, j) - sum + e[j]);
}
    """ % dict(eps=eps)
    local_dict = dict(L=L, n=n, e=e, A=A, norm_A=norm_A,
                      gamm=gamm, delta=delta)
    headers = ['<math.h>', '<algorithm>']
    scipy.weave.inline(code,
                    local_dict.keys(),
                    local_dict=local_dict,
                    headers=headers)
    return (L, e)


def chol_gko(A, eps=_EPS):
    """
Modified Cholesky decomposition
using the Gill-King rutine
Implemented by M. L. Overton

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
>>> L, e = chol_gmo(A)
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
#      indef = 0

    # initialize d and L

    d = np.zeros(n, dtype=float)
    if sparse.issparse(A):
        L = sparse.eye(*A.shape)
    else:
        L = np.eye(n);

    # there are no inner for loops, everything implemented with
    # vector operations for a reasonable level of efficiency

    for j in xrange(n):
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
#          if d[j] > djtemp:  # A was not sufficiently positive definite
#              indef = True

    # convert to usual output format: replace L by L*sqrt(D) and transpose
    for j in xrange(n):
        L[:, j] = L[:, j]*np.sqrt(d[j]) # L = L*diag(sqrt(d)) bad in sparse case

    e = (np.dot(L, L.T) - A).diagonal()

    return L, e


def chol_se(A, eps=_EPS):
    """
A revised modified Cholesky factorisation algorithm.

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
>>> P, L, e = chol_se(A)
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
            for j in xrange(k+1, n-2):
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

    for j in xrange(n):

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


def chol_bo(A, eps=_EPS):
    """
Pivoting Cholesky Decompostion
using algorithm by Bastos-O'Hagan

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
>>> P, L, E = chol_bo(A)
>>> P, L = np.matrix(P), np.matrix(L)
>>> print(np.diag(E))
[ -8.88178420e-16   0.00000000e+00   2.33733333e+00]
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

    for k in xrange(n):

        q = np.argmax( np.diag(A[k:, k:]) ) + k

        if A[q, q] <= np.abs(a0*eps):

            if not k:
                raise ValueError("Purly negative definite")

            for j in xrange(k, n):
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
    for k in xrange(n):
        P[k, piv[k]] = 1.

    E = np.dot(R.T, R) - np.dot(np.dot(P.T, B), P)

    return P, R.T, E


if __name__=="__main__":
    import doctest
    doctest.testmod()
