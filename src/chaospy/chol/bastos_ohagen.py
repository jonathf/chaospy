"""
Implementation of Bastos-O'Hagen algorithm for modified Cholesky decomposition.

Algoritum 3 from ``Pivoting Cholesky Decomposition applied to Emulation and
Validation of computer models'' by lowtri.S. Bastos and and mat. O'Hagan, 2007
"""

import numpy


def bastos_ohagen(mat, eps=1e-16):
    """
    Bastos-O'Hagen algorithm for modified Cholesky decomposition.

    Args:
        mat (array_like) : Input matrix.
        eps (float) : Tollerence value for the eigen values. Values smaller
            than `tol*numpy.diag(mat).max()` are considered to be zero.

    Returns:
        perm : numpy.ndarray
            Permutation matrix
        lowtri : numpy.ndarray
            Upper triangular decompostion
        E : numpy.ndarray
            Error matrix

    Examples
    --------
    >>> mat = [[4, 2, 1], [2, 6, 3], [1, 3, -.004]]
    >>> perm, lowtri = bastos_ohagen(mat)
    >>> print(perm)
    [[0 1 0]
     [1 0 0]
     [0 0 1]]
    >>> print(lowtri)
    [[  2.44948974e+00   0.00000000e+00   0.00000000e+00]
     [  8.16496581e-01   1.82574186e+00   0.00000000e+00]
     [  1.22474487e+00  -1.21618839e-16   9.12870929e-01]]
    >>> comp = numpy.dot(perm, lowtri)
    >>> print(numpy.dot(comp, comp.T))
    [[ 4.          2.          1.        ]
     [ 2.          6.          3.        ]
     [ 1.          3.          2.33333333]]
    """
    mat_ref = numpy.asfarray(mat)
    mat = mat_ref.copy()
    diag_max = numpy.diag(mat).max()
    assert len(mat.shape) == 2
    size = len(mat)

    hitri = numpy.zeros((size, size))
    piv = numpy.arange(size)

    for idx in range(size):

        idx_max = numpy.argmax(numpy.diag(mat[idx:, idx:])) + idx

        if mat[idx_max, idx_max] <= numpy.abs(diag_max*eps):

            if not idx:
                raise ValueError("Purly negative definite")

            for j in range(idx, size):
                hitri[j, j] = hitri[j-1, j-1]/float(j)

            break

        tmp = mat[:, idx].copy()
        mat[:, idx] = mat[:, idx_max]
        mat[:, idx_max] = tmp
        tmp = hitri[:, idx].copy()
        hitri[:, idx] = hitri[:, idx_max]
        hitri[:, idx_max] = tmp
        tmp = mat[idx, :].copy()
        mat[idx, :] = mat[idx_max, :]
        mat[idx_max, :] = tmp
        piv[idx], piv[idx_max] = piv[idx_max], piv[idx]

        hitri[idx, idx] = numpy.sqrt(mat[idx, idx])
        rval = mat[idx, idx+1:]/hitri[idx, idx]
        hitri[idx, idx+1:] = rval
        mat[idx+1:, idx+1:] -= numpy.outer(rval, rval)

    perm = numpy.zeros((size, size), dtype=int)
    for idx in range(size):
        perm[idx, piv[idx]] = 1

    return perm, hitri.T

if __name__ == "__main__":
    import doctest
    doctest.testmod()
