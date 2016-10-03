"""
Use of Bertran to calculate sparse segments.
"""
import numpy as np
import chaospy as cp


def sparse_segment(cords):
    r"""
    Create a segment of a sparse grid.

    More specifically, a segment of
    `\cup_{cords \in C} sparse_segment(cords) == sparse_grid(M)`
    where
    `C = {cords: M=sum(cords)}`

    Parameters
    ----------
    cords : array_like
        The segment to extract. `cord` must consist of non-negative intergers.

    Returns
    -------
    Q : ndarray
        Sparse segment where `Q.shape==(K, sum(M))` and `K` is segment
        specific.

    Convert a ol-index to sparse grid coordinates on [0, 1]^N hyper
    cube. A sparse grid of order `D` coencide with the set of
    sparse_segments where `||cords||_1 <= D`.

    Examples
    --------
    >>> cp.sparse_segment([0, 2])
    array([[ 0.5  ,  0.125],
           [ 0.5  ,  0.375],
           [ 0.5  ,  0.625],
           [ 0.5  ,  0.875]])

    >>> cp.sparse_segment([0, 1, 0, 0])
    array([[ 0.5 ,  0.25,  0.5 ,  0.5 ],
           [ 0.5 ,  0.75,  0.5 ,  0.5 ]])
    """
    cords = np.array(cords)+1
    slices = []
    for cord in cords:
        slices.append(slice(1, 2**cord+1, 2))

    grid = np.mgrid[slices]
    indices = grid.reshape(len(cords), np.prod(grid.shape[1:])).T
    sgrid = indices*2.**-cords
    return sgrid
