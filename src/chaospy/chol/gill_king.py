"""
Algorithm 3.4 of 'Numerical Optimization' by Jorge Nocedal and Stephen J.
Wright

This is based on the MATLAB code from Michael L. Overton <overton@cs.nyu.edu>:
http://cs.nyu.edu/overton/g22_opt/codes/cholmod.m
"""

import numpy
import scipy.sparse


def gill_king(mat, eps=1e-16):
    """
    Gill-King algorithm for modified cholesky decomposition.

    Args:
        mat (numpy.ndarray) : Must be a non-singular and symmetric matrix.  If
            sparse, the result will also be sparse.
        eps (float) : Error tolerance used in algorithm.


    Returns:
        lowtri (numpy.ndarray) : Lower triangular Cholesky factor.

    Examples:
        >>> mat = [[4, 2, 1], [2, 6, 3], [1, 3, -.004]]
        >>> lowtri = gill_king(mat)
        >>> print(lowtri)
        [[ 2.          0.          0.        ]
         [ 1.          2.23606798  0.        ]
         [ 0.5         1.11803399  1.22637678]]
        >>> print(numpy.dot(lowtri, lowtri.T))
        [[ 4.     2.     1.   ]
         [ 2.     6.     3.   ]
         [ 1.     3.     3.004]]
    """
    if not scipy.sparse.issparse(mat):
        mat = numpy.asfarray(mat)
    assert numpy.allclose(mat, mat.T)

    size = mat.shape[0]
    mat_diag = mat.diagonal()
    gamma = abs(mat_diag).max()
    off_diag = abs(mat - numpy.diag(mat_diag)).max()

    delta = eps*max(gamma + off_diag, 1)
    beta = numpy.sqrt(max(gamma, off_diag/size, eps))

    lowtri = _gill_king(mat, beta, delta)

    return lowtri


def _gill_king(mat, beta, delta):
    """Backend function for the Gill-King algorithm."""
    size = mat.shape[0]

    # initialize d_vec and lowtri
    if scipy.sparse.issparse(mat):
        lowtri = scipy.sparse.eye(*mat.shape)
    else:
        lowtri = numpy.eye(size)

    d_vec = numpy.zeros(size, dtype=float)

    # there are no inner for loops, everything implemented with
    # vector operations for a reasonable level of efficiency
    for idx in range(size):
        if idx == 0:
            idz = []     # column index: all columns to left of diagonal
                        # d_vec(idz) doesn't work in case idz is empty
        else:
            idz = numpy.s_[:idx]

        djtemp = mat[idx, idx] - numpy.dot(
            lowtri[idx, idz], d_vec[idz]*lowtri[idx, idz].T)
        # C(idx, idx) in book

        if idx < size - 1:
            idy = numpy.s_[idx+1:size]
            # row index: all rows below diagonal
            ccol = mat[idy, idx] - numpy.dot(
                lowtri[idy, idz], d_vec[idz]*lowtri[idx, idz].T)
            # C(idy, idx) in book
            theta = abs(ccol).max()
            # guarantees d_vec(idx) not too small and lowtri(idy, idx) not too
            # big in sufficiently positive definite case, d_vec(idx) = djtemp
            d_vec[idx] = max(abs(djtemp), (theta/beta)**2, delta)
            lowtri[idy, idx] = ccol/d_vec[idx]

        else:
            d_vec[idx] = max(abs(djtemp), delta)

    # convert to usual output format: replace lowtri by lowtri*sqrt(D) and
    # transpose
    for idx in range(size):
        lowtri[:, idx] = lowtri[:, idx]*numpy.sqrt(d_vec[idx])
        # lowtri = lowtri*diag(sqrt(d_vec)) bad in sparse case

    return lowtri


    if __name__ == "__main__":
        import doctest
        doctest.testmod()
