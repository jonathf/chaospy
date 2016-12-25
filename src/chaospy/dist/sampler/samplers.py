"""
Functions for generating samples on the unite hyper cube.
"""

import numpy as np

import chaospy.quad

def chebyshev(dim, n):
    "Chebyshev sampling"
    x = .5*np.cos(np.arange(n,0,-1)*np.pi/(n+1)) + .5
    X = chaospy.quad.combine([x]*dim)
    return X.T


def chebyshev_nested(dim, n):
    "Chebyshev sampling, nested"
    x = .5*np.cos(np.arange(2**n-1,0,-1)*np.pi/(2**n)) + .5
    X = chaospy.quad.combine([x]*dim)
    return X.T


def regular_grid(dim, n):
    "regular grid"
    x = np.arange(1,n+1)/(n+1.)
    X = chaospy.quad.combine([x]*dim)
    return X.T


def regular_grid_nested(dim, n):
    "regular grid, nested"
    x = np.arange(1,2**n)*1./(2**n)
    X = chaospy.quad.combine([x]*dim)
    return X.T

def corput(p, k):
    """
Van der Corput sampling
    """
    k += 1
    out = 0.
    p_base = p
    while k > 0:
        a = k % p_base
        out += a*1./p
        k = int(k / p_base)
        p *= p_base
    return out


def primes(n):
    """
    Generate primes using sieve of Eratosthenes.

    Examples:
        >>> print(primes(20))
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if n == 2:
        return [2]

    elif n < 2:
        return []

    s = list(range(3, n+1, 2))
    mroot = n ** 0.5
    half = int((n+1)/2-1)
    i = 0
    m = 3
    while m <= mroot:
        if s[i]:
            j = int((m*m-3)/2)
            s[j] = 0
            while j < half:
                s[j] = 0
                j += m
        i = i+1
        m = 2*i+3
    out = [2] + [x for x in s if x]
    return out


def hammersley(dim, n):
    """
    Hammersley sequence.

    Examples:
        >>> print(hammersley(2, 4))
        [[ 0.2    0.4    0.6    0.8  ]
         [ 0.5    0.25   0.75   0.125]]
    """
    p = []
    m = 10
    while len(p)<dim:
        p = primes(m)
        m *= 2
    p = p[:dim]

    out = np.zeros((dim, n))
    out[0] = np.arange(1, n+1)*1./(n+1)
    for i in range(1, dim):
        for j in range(n):
            out[i,j] = corput(p[i-1], j)
    return out


def halton(dim, n):
    """
    Halton sequence.

    Examples:
        >>> print(halton(2, 4))
        [[ 0.125       0.625       0.375       0.875     ]
         [ 0.44444444  0.77777778  0.22222222  0.55555556]]
    """
    p = []
    m = 10
    while len(p) < dim:
        p = primes(m)
        m *= 2

    p = p[:dim]
    burn = p[-1]

    out = np.empty((n, dim))
    for i in range(n):
        for j in range(dim):
            out[i,j] = corput(p[j], i+burn)
    return out.T


def korobov(dim, n, a=17797):
    """
    Korobov set.

    Examples:
        >>> print(korobov(2, 4))
        [[ 0.2  0.4  0.6  0.8]
         [ 0.4  0.8  0.2  0.6]]
    """
    z = np.empty(dim)
    z[0] = 1
    for i in range(1,dim):
        z[i] = a*z[i-1] % (n+1)

    grid = np.mgrid[:dim,:n+1]
    Z = z[grid[0]]
    B = grid[1]+1
    P = B*Z/(n+1) % 1
    return P[:,:n]


def latin_hypercube(dim, n):
    """
    Latin Hypercube sampling.

    Examples:
        >>> cp.seed(1000)
        >>> print(cp.latin_hypercube(2, 4))
        [[ 0.6633974   0.27875174  0.98757072  0.12054785]
         [ 0.46811863  0.05308317  0.51017741  0.84929862]]
    """
    R = np.random.random(n*dim).reshape((dim, n))
    for d in range(dim):
        perm = np.random.permutation(n)
        R[d] = (perm + R[d])/n
    return R

if __name__ == "__main__":
    import chaospy as cp
    import doctest
    doctest.testmod()
