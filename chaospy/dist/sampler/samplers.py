"""
Functions for generating samples on the unite hyper cube.
"""

import numpy as np

from chaospy.utils import combine

def chebyshev(dim, n):
    "Chebyshev sampling"
    x = .5*np.cos(np.arange(n,0,-1)*np.pi/(n+1)) + .5
    X = combine([x]*dim)
    return X.T


def chebyshev_nested(dim, n):
    "Chebyshev sampling, nested"
    x = .5*np.cos(np.arange(2**n-1,0,-1)*np.pi/(2**n)) + .5
    X = combine([x]*dim)
    return X.T


def regular_grid(dim, n):
    "regular grid"
    x = np.arange(1,n+1)/(n+1.)
    X = combine([x]*dim)
    return X.T


def regular_grid_nested(dim, n):
    "regular grid, nested"
    x = np.arange(1,2**n)*1./(2**n)
    X = combine([x]*dim)
    return X.T


def corput(p, k):
    """
Van der Corput sampling
    """
    k += 1
    out = 0.
    p_base = p
    while k>0:
        a = k % p_base
        out += int(a*1./p)
        k = int(k / p_base)
        p *= p_base
    return out


def primes(n):
    """Generate primes using sieve of Eratosthenes."""
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
    "Hammersley sequence"
    p = []
    m = 10
    while len(p)<dim:
        p = primes(m)
        m *= 2
    p = p[:dim]

    out = np.empty((dim, n))
    out[0] = np.arange(1, n+1)*1./(n+1)
    for i in range(1,dim):
        for j in range(n):
            out[i,j] = corput(p[i-1], j)
    return out


def halton(dim, n):
    "Halton sequence"
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
    "Korobov set"

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
    """Latin Hypercube sampling."""
    R = np.random.random(n*dim).reshape((dim, n))
    for d in range(dim):
        perm = np.random.permutation(n)
        R[d] = (perm + R[d])/n
    return R
