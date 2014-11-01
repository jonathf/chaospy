"""
Multi-indexing tools after Bertran's notation

Polynomials can either be indexed with single index or multi-index
Bertran is a notation for switching between the two.

Functions
---------
add             Addition of two intergers in Bertran's notation
multi_index     Convert from single-index to multi-indices
single_index    Convert from multi-indices to single-index
terms           Number of terms in an expansion
parent          Find parent index
child           Find child index
olindex         All multi-indices of a given order
rank            Calculate the index rank
"""

import numpy as np
from scipy.misc import comb
from utils import lazy_eval

__version__ = "1.0"
__all__ = [
"add",
"multi_index",
"single_index",
"terms",
"parent",
"child",
"olindex",
"rank",
"sparse_segment",
"bindex",
]


def add(i, j, dim):
    """
Bertran addition

Example
-------
>>> print add(3, 3, 1)
6
>>> print add(3, 3, 2)
10
"""
    I = np.array(multi_index(i, dim))
    J = np.array(multi_index(j, dim))
    out = single_index(I+J)
    return out

def terms(M, dim):
    """
Count the number of polynomials in an expansion.

Parameters
----------
M : int
    The upper order for the expansion.
dim : int
    The number of dimensions of the expansion.

Returns
-------
N : int
    The number of terms in an expansion of upper order `M` and
    number of dimensions `dim`.
    """
    return int(comb(M+dim,dim, 1))


def multi_index(i, dim):
    """
Single to multi-index using graded reverse lexicographical
notation.

Parameters
----------
i : int
    Index in interger notation
dim : int
    The number of dimensions in the multi-index notation

Returns
-------
I : tuple
    Multi-index of `i` with `len(I)=dim`

Examples
--------
>>> for i in range(5):
...     print cp.multi_index(i,3)
(0, 0, 0)
(1, 0, 0)
(0, 1, 0)
(0, 0, 1)
(2, 0, 0)

See Also
--------
single_index
    """

    def _rec(i, D):
        N, M = 0, 0
        if not D:
            return ()

        if i==0:
            return (0,)*D

        while terms(N,D)<=i: N+=1
        i -= terms(N-1,D)

        if i==0: return (N,) + (0,)*(D-1)
        while terms(M,D-1)<=i: M+=1

        return (int(N-M),) + _rec(i, D-1)

    return _rec(i, dim)




def bindex(start, stop=None, dim=1, sort="G"):
    """
Generator for creating multi-indices

Parameters
----------
start : int
    The lower order of the indices
stop : int, optional
    the maximum shape included. If omitted:
    stop <- start; start <- 0
    If int is provided, set as largest total order.
    If array of int, set as largest order along each axis.
dim : int
    The number of dimensions in the expansion

Returns
-------
indices : list
    Grevlex order list of indices.

Examples
--------
>>> print cp.bindex(0, 1, 3)
[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
>>> print cp.bindex(2, 3, 2)
[(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    """

    if stop==None:
        start, stop = 0, start
    sort = sort.upper()

    lo = single_index([start] + [0]*(dim-1))
    up = single_index([0]*(dim-1) + [start])

    local, total = [], []
    for i in xrange(lo, up+1):
        I = multi_index(i, dim)
        local.append(list(I))
        total.append(I)

    for m in xrange(start, stop):

        local_, local = local, []
        for I in local_:

            i = 0
            for i in xrange(len(I)):
                if sum(I[i+1:])==0:
                    break

            for j in xrange(i, len(I)):
                I[j]+=1
                local.append(I[:])
                total.append(tuple(I))
                I[j]-=1

    if "G" in sort:
        _cmp = lambda i,j: cmp(sum(i),sum(j))
    else:
        def _cmp(i,j):
            if not np.any(i): return 0
            return cmp(i[-1], j[-1]) or _cmp(i[:-1], j[:-1])
    total.sort(cmp=_cmp)

    if "I" in sort:
        total = total[::-1]

    if "R" in sort:
        total = [I[::-1] for I in total]

    return total



def single_index(I):
    """
Multi-index to single integer notation using graded reverse
lexicographical notation.

Parameters
----------
I : array_like
    Index in multi-index notation

Returns
i : int
    Integer index of `I`

Examples
--------
>>> for i in range(3):
...     print single_index(np.eye(3)[i])
1
2
3

    """
    if -1 in I:
        return 0
    N, D = int(sum(I)), len(I)
    if N==0: return 0
    return terms(N-1,D) + single_index(I[1:])



def rank(i, dim):
    """
Calculate the index rank according to Bertran's notation.
    """

    I = multi_index(i, dim)
    rank = 0
    while I[-1:]==(0,):
        rank += 1
        I = I[:-1]
    return rank

def parent(i, dim, ax=None):
    """
Parent node according to Bertran's notation.

Parameters
----------
i : int
    Index of the child node.
dim : int
    Dimensionality of the problem.

Returns
-------
j : int
    Index of parent node with `j<=i`, and `j==i` iff `i==0`.
ax : int
    Dimension direction the parent was found.

    """

    I = multi_index(i, dim)
    if ax is None:
        ax = dim - np.argmin(1*(np.array(I)[::-1]==0))-1

    if not i: return i, ax

    if I[ax]==0:
        j = parent(parent(i, dim)[0], dim)[0]
        while child(j+1, dim, ax)<i: j += 1
        return j, ax

    out = np.array(I) - 1*(np.eye(dim)[ax])
    return single_index(out), ax

parent = lazy_eval(parent, int)


def child(i, dim, ax):
    """
Child node according to Bertran's notation.

Parameters
----------
i : int
    Index of the parent node.
dim : int
    Dimensionality of the problem.
ax : int
    Dimension direction to define a child.
    Must have `0<=ax<dim`

Returns
-------
j : int
    Index of child node with `j>i`.

Examples
--------
>>> print cp.child(4, 1, 0)
5
>>> print cp.child(4, 2, 1)
8
    """
    I = multi_index(i, dim)
    out = np.array(I) + 1*(np.eye(len(I))[ax])
    return single_index(out)

child = lazy_eval(child, int)


def olindex(order, dim):
    """
Create an lexiographical sorted basis for a given order.

Examples
--------
>>> cp.olindex(3,2)
array([[0, 3],
       [1, 2],
       [2, 1],
       [3, 0]])
    """

    A = [0 for i in range(dim)]
    out = []

    def rec_call(I):

        if np.sum(A)==order:
            out.append(A[:])
            return

        if I==dim:
            return

        sA = np.sum(A)
        a = A[I]

        for i in xrange(order-np.sum(A)+1):

            A[I] = i

            if sA<order:
                rec_call(I+1)

            else:
                break
        A[I] = a

    rec_call(0)
    return np.array(out)


def sparse_segment(cords):
    """
A segmentation of a sparse grid
`\cup_{cords \in C} sparse_segment(cords)==sparse_grid(M)` where
`C = {cords: M=sum(cords)}`

Parameters
----------
cords : array_like
    The segment to extract. `cord` must consist of non-negative
    intergers.

Returns
-------
Q : ndarray
    Sparse segment where `Q.shape==(K,sum(M))` and `K` is segment
    specific.

Convert a ol-index to sparse grid coordinates on [0,1]^N hyper
cube. A sparse grid of order `D` coencide with the set of
sparse_segments where `||cords||_1 <= D`.

Examples
--------
>>> cp.sparse_segment([0,2])
array([[ 0.5  ,  0.125],
       [ 0.5  ,  0.375],
       [ 0.5  ,  0.625],
       [ 0.5  ,  0.875]])

>>> cp.sparse_segment([0,1,0,0])
array([[ 0.5 ,  0.25,  0.5 ,  0.5 ],
       [ 0.5 ,  0.75,  0.5 ,  0.5 ]])
    """

    cords = np.array(cords)+1
    slices = []
    for cord in cords:
        slices.append(slice(1,2**cord+1,2))

    grid = np.mgrid[slices]
    indices = grid.reshape(len(cords), np.prod(grid.shape[1:])).T
    sgrid = indices*2.**-cords
    return sgrid

class Fourier_recursive(object):
    """
Calculate Fourier coefficients using Bertran's recursive formula.
    """

    def __init__(self, dist):
        """
Distribution to create orthogonal basis on

coef = E[v[n]*P[i]*P[j]]/E[P[j]**2]
where v is basis polynomial and P are orthogonal polynomials
        """

        self.dist = dist
        self.hist = {}

    def __call__(self, n, i, j):
        """
n : int
    Single index for basis reference
i : int
    Single index for for orthogonal poly in nominator
j : int
    Single index for for orthogonal poly in denominator
        """

        if (n,i,j) in self.hist:
            return self.hist[n,i,j]

        dim = len(self.dist)

        if n==j==0 or n==i==0:
            out = 0

        elif add(n,j,dim)<i or add(n,i,dim)<j:
            out = 0

        elif add(n,i,dim)==j:
            out = 1

        elif i==j==0:
            out = self.dist.mom(multi_index(n, dim))

        elif j==0:

            rank_ = min(rank(n,dim), rank(i,dim), rank(j,dim))
            par, ax0 = parent(i, dim)
            gpar, ax1 = parent(par, dim, ax0)
            dn = child(n, dim, ax0)
            oneup = child(0, dim, ax0)

            out = self(dn, par, 0)
            for k in xrange(gpar, i):
                if rank(k,dim)>=rank_:
                    A = self(oneup, par, k)
                    B = self(n, k, 0)
                    out = out - A*B

        else:

            rank_ = min(rank(n,dim), rank(i,dim), rank(j,dim))
            par, ax0 = parent(j, dim)
            gpar, ax1 = parent(par, dim, ax0)
            dn = child(n, dim, ax0)
            oneup = child(0, dim, ax0)
            twoup = child(oneup, dim, ax0)

            out1 = self(dn, i, par)
            out2 = self(twoup, par, par)
            for k in xrange(gpar, j):
                if rank(k,dim)>=rank_:
                    A = self(oneup, k, par)
                    B = self(n, i, k)
                    out1 = out1 - A*B
                    C = self(oneup, par, k)
                    D = self(oneup, k, par)
                    out2 = out2 - C*D
            out = out1 / out2

        self.hist[n,i,j] = out

        return out



if __name__=="__main__":
    import __init__ as cp
    import doctest
    doctest.testmod()

