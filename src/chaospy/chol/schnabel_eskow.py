"""
Implementation of Schnabel-Eskow modified cholesky factorisation.

Schnabel, R. B. and Eskow, E. 1999, mat revised modifed cholesky factorisation
algorithm. SIAM J. Optim. 9, 1135-1148.
"""

import numpy


def schnabel_eskow(mat, eps=1e-16):
    """
    Scnabel-Eskow algorithm for modified Cholesky factorisation algorithm.

    Args:
        mat (numpy.ndarray) : Must be a non-singular and symmetric matrix If
            sparse, the result will also be sparse.
        eps (float) : Error tolerance used in algorithm.

    Returns
    -------
    perm : 2d array
        Permutation matrix used for pivoting.
    lowtri : 2d array
        Lower triangular factor
    err : 1d array
        Positive diagonals of shift matrix `err`.

    Examples
    --------
    >>> mat = [[4, 2, 1], [2, 6, 3], [1, 3, -.004]]
    >>> perm, lowtri, err = schnabel_eskow(mat)
    >>> perm, lowtri = numpy.matrix(perm), numpy.matrix(lowtri)
    >>> print(perm*lowtri*lowtri.T*perm.T)
    [[ 5.50402245  2.          1.        ]
     [ 2.          6.          3.        ]
     [ 1.          3.          1.50002245]]
    """
    mat = numpy.asfarray(mat)
    tau = eps**(1/3.)

    # Create the matrix err and mat.
    size = mat.shape[0]
    mat0 = mat
    mat = 1*mat
    err = numpy.zeros(size, dtype=float)

    # Permutation matrix.
    perm = numpy.eye(size)

    # Calculate gamma.
    gamma = abs(mat.diagonal()).max()

    # Phase one, mat potentially positive definite.
    ###############################################

    def invariant(mat, k):
        """Return `True` if the invariant is satisfied."""
        A_ = numpy.eye(size)
        L_ = numpy.eye(size)
        A_[k:, k:] = numpy.triu(mat[k:, k:], 0) + numpy.triu(mat[k:, k:], 1).T
        L_[:, :k] = numpy.tril(mat[:, :k])
        return numpy.allclose(
            numpy.dot(L_, numpy.dot(A_, L_.T)),
            numpy.dot(perm, numpy.dot(mat0, perm.T)) + numpy.diag(err)
        )

    def jiter_factor(mat, j, perm, err):
        """Perform jth iteration of factorisation.
        """
        assert invariant(mat, j)

        mat[j, j] = numpy.sqrt(mat[j, j])
        mat[j+1:, j] /= mat[j, j]
        mat[j+1:, j+1:] -= mat[j+1:, j:j+1]*mat[j+1:, j:j+1].T
        mat[j, j+1:] = 0

        assert invariant(mat, j+1)

    def permute(mat, perm, i, j):
        """Exchange rows and columns i and j of mat and recored the
        permutation in perm"""
        p = numpy.arange(size, dtype=int)
        if i != j:
            p[[i, j]] = j, i

            perm[::] = perm[p, :]
            mat[::] = mat[p, :]
            mat[::] = mat[:, p]

    def exec_phasetwo(mat, perm, err, j):
        """Phase 2 of the algorithm, mat not positive definite."""
        if j == size:
            # delta = err[size].
            delta = -mat[size-1, size-1] + max(
                -tau*mat[size-1, size-1]/(1 - tau), tau*tau*gamma)

            err[size-1] = delta
            mat[size-1, size-1] += delta
            mat[size-1, size-1] = numpy.sqrt(mat[size-1, size-1])

        else:
            # Number of iterations performed in phase one (less 1).
            k = j - 1

            # Calculate the lower Gerschgorin bounds of Ak+1.
            tmp = mat[k+1:, k+1:]
            g = tmp.diagonal().copy()
            tmp = abs(numpy.tril(tmp, -1))
            g -= tmp.sum(axis=0)
            g -= tmp.sum(axis=1)

            # Modified Cholesky decomposition.
            delta_prev = 0.0
            for j in range(k+1, size-2):
                # Pivot on the maximum lower Gerschgorin bound
                # estimate.
                i = j + numpy.argmax(g[j-(k+1):])

                # Interchange row and column i and j.
                permute(mat, perm, i, j)

                # Calculate err[j] and add to diagonal.
                normj = abs(mat[j+1:, j]).sum()
                delta = max(0.0,
                            -mat[j, j] + max(normj, tau*tau*gamma),
                            delta_prev)  # delta = E[size].

                if delta > 0.0:
                    mat[j, j] += delta
                    delta_prev = delta
                err[j] = delta

                # Update Gerschgorin bound estimates.
                if mat[j, j] != normj:
                    temp = 1.0 - normj / mat[j, j]
                    g[j-k:] += abs(mat[j+1:, j])*temp

                # Perform jth iteration of factorisation.
                jiter_factor(mat, j, perm, err)

            # Final 2*2 submatrix.
            mini = mat[-2:, -2:]
            mini[1, 0] = mini[0, 1]
            eigs = numpy.sort(numpy.linalg.eigvalsh(mini))
            delta = max(
                0, -eigs[0] + max(tau*(eigs[1] - eigs[0])/(1 - tau),
                                  tau*tau*gamma), delta_prev)
            if delta > 0.0:
                mat[size-2, size-2] += delta
                mat[size-1, size-1] += delta
                delta_prev = delta
            err[size-2] = err[size-1] = delta

            mat[size-2, size-2] = numpy.sqrt(mat[size-2, size-2])
            mat[size-1, size-2] /= mat[size-2, size-2]
            mat[size-1, size-1] = numpy.sqrt(
                mat[size-1, size-1] - mat[size-1, size-2]**2)

    for j in range(size):

        # Calculate max_Aii and min_Aii
        diag = mat.diagonal()[j:]

        # Test for phase 2, mat not positive definite.
        if diag.max() < tau*tau * gamma or diag.min() < - 0.1 * diag.max():
            exec_phasetwo(mat, perm, err, j)
            break
        else:
            # Pivot on maximum diagonal of remaining submatrix.
            i = j + numpy.argmax(mat.diagonal()[j:])

            # Interchange row and column i and j.
            permute(mat, perm, i, j)

            # Test for phase 2 again.
            min_num = 1e99
            mat_diag = mat.diagonal()
            if j + 1 < size:
                min_num = (mat_diag[i] -
                           mat[i, j+1:]**2/mat_diag[j+1:]).min()
            else:
                min_num = mat_diag[i]

            if j+1 <= size and min_num < - 0.1 * gamma:
                exec_phasetwo(mat, perm, err, j)
                break

            # Perform jth iteration of factorisation.
            else:
                jiter_factor(mat, j, perm, err)

    # The Cholesky factor of mat.
    return perm, numpy.tril(mat), err

if __name__ == "__main__":
    import doctest
    doctest.testmod()
