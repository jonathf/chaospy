"""
Implementation of the Gill-Murray-Wright modified Cholesky algorithm.

Algorithm 6.5 from page 148 of 'Numerical Optimization' by Jorge Nocedal and
Stephen J. Wright, 1999, 2nd ed.
"""

import numpy


def gill_murray_wright(mat, eps=1e-16):
    """
    Gill-Murray-Wright algorithm for pivoting modified Cholesky decomposition.

    Return `(perm, lowtri, error)` such that
    `perm.T*mat*perm = lowtri*lowtri.T` is approximately correct.

    Args:
        mat (array) : Must be a non-singular and symmetric matrix
        eps (float) : Error tolerance used in algorithm.

    Returns:
        (numpy.ndarray, numpy.ndarray) : Permutation matrix used for pivoting
            and lower triangular factor.

    Examples:
        >>> mat = numpy.matrix([[4, 2, 1], [2, 6, 3], [1, 3, -.004]])
        >>> perm, lowtri = gill_murray_wright(mat)
        >>> perm, lowtri = numpy.matrix(perm), numpy.matrix(lowtri)
        >>> print(perm)
        [[0 1 0]
         [1 0 0]
         [0 0 1]]
        >>> print(lowtri)
        [[  2.44948974e+00   0.00000000e+00   0.00000000e+00]
         [  8.16496581e-01   1.82574186e+00   0.00000000e+00]
         [  1.22474487e+00  -1.21618839e-16   1.22637678e+00]]
        >>> print(perm*lowtri*lowtri.T*perm.T)
        [[ 4.     2.     1.   ]
         [ 2.     6.     3.   ]
         [ 1.     3.     3.004]]
    """
    mat = numpy.asfarray(mat)
    size = mat.shape[0]

    # Calculate gamma(mat) and xi_(mat).
    gamma = 0.0
    xi_ = 0.0
    for idy in range(size):
        gamma = max(abs(mat[idy, idy]), gamma)
        for idx in range(idy+1, size):
            xi_ = max(abs(mat[idy, idx]), xi_)

    # Calculate delta and beta.
    delta = eps * max(gamma + xi_, 1.0)
    if size == 1:
        beta = numpy.sqrt(max(gamma, eps))
    else:
        beta = numpy.sqrt(max(gamma, xi_ / numpy.sqrt(size*size - 1.0), eps))

    # Initialise data structures.
    mat_a = 1.0 * mat
    mat_r = 0.0 * mat
    perm = numpy.eye(size, dtype=int)

    # Main loop.
    for idx in range(size):

        # Row and column swapping, find the index > idx of the largest
        # idzgonal element.
        idz = idx
        for idy in range(idx+1, size):
            if abs(mat_a[idy, idy]) >= abs(mat_a[idz, idz]):
                idz = idy

        if idz != idx:
            mat_a, mat_r, perm = swap_across(idz, idx, mat_a, mat_r, perm)

        # Calculate a_pred.
        theta_j = 0.0
        if idx < size-1:
            for idy in range(idx+1, size):
                theta_j = max(theta_j, abs(mat_a[idx, idy]))
        a_pred = max(abs(mat_a[idx, idx]), (theta_j/beta)**2, delta)

        # Calculate row idx of r and update a.
        mat_r[idx, idx] = numpy.sqrt(a_pred)
        for idy in range(idx+1, size):
            mat_r[idx, idy] = mat_a[idx, idy] / mat_r[idx, idx]
            for idz in range(idx+1, idy+1):

                # Keep matrix a symmetric:
                mat_a[idy, idz] = mat_a[idz, idy] = \
                    mat_a[idz, idy] - mat_r[idx, idy] * mat_r[idx, idz]

    # The Cholesky factor of mat.
    return perm, mat_r.T


def swap_across(idx, idy, mat_a, mat_r, perm):
    """Interchange row and column idy and idx."""
    # Temporary permutation matrix for swaping 2 rows or columns.
    size = mat_a.shape[0]
    perm_new = numpy.eye(size, dtype=int)

    # Modify the permutation matrix perm by swaping columns.
    perm_row = 1.0*perm[:, idx]
    perm[:, idx] = perm[:, idy]
    perm[:, idy] = perm_row

    # Modify the permutation matrix p by swaping rows (same as
    # columns because p = pT).
    row_p = 1.0 * perm_new[idx]
    perm_new[idx] = perm_new[idy]
    perm_new[idy] = row_p

    # Permute mat_a and r (p = pT).
    mat_a = numpy.dot(perm_new, numpy.dot(mat_a, perm_new))
    mat_r = numpy.dot(mat_r, perm_new)
    return mat_a, mat_r, perm


if __name__ == "__main__":
    import doctest
    doctest.testmod()
