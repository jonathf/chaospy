"""
Set of methods for generating samples on the unit hypercube, and
map them to a distribution domain as a sample scheme.

See samplegen for up-to-date frontend.

sample_select and Sampler are obsolete and only available as legacy

"""

import numpy as np
from .sobol_lib import sobol

from chaospy.utils import combine
from chaospy import bertran as be
from chaospy.quadrature import golub_welsch as _gw, gauss_legendre


def samplegen(order, domain, rule="S", antithetic=None,
        verbose=False):
    """
Sample generator

Parameters
----------
order : int
    Sample order
domain : Dist, int, array_like
    Defines the space where the samples are generated.
    Dist
        Mapped to distribution domain using inverse Rosenblatt.
    int
        No mapping, but sets the number of dimension.
    array_like
        Stretch samples such that they are in [domain[0], domain[1]]
rule : str
    rule for generating samples, where d is the number of
    dimensions.

    Key     Name                Nested
    ----    ----------------    ------
    "C"     Chebyshev nodes     no
    "NC"    Nested Chebyshev    yes
    "G"     Gaussian quadrature no
    "K"     Korobov             no
    "R"     (Pseudo-)Random     no
    "RG"    Regular grid        no
    "NG"    Nested grid         yes
    "L"     Latin hypercube     no
    "S"     Sobol               yes
    "H"     Halton              yes
    "M"     Hammersley          yes

antithetic : array_like, optional
    List of bool. Represents the axes to mirror using antithetic
    variable.
    """


    rule = rule.upper()

    if isinstance(domain, int):
        dim = domain
        trans = lambda x, verbose:x

    elif isinstance(domain, (tuple, list, np.ndarray)):
        domain = np.asfarray(domain)
        if len(domain.shape)<2:
            dim = 1
        else:
            dim = len(domain[0])
        lo,up = domain
        trans = lambda x, verbose: ((up-lo)*x.T + lo).T

    else:
        dist = domain
        dim = len(dist)
        trans = dist.inv

    if not (antithetic is None):

        antithetic = np.array(antithetic, dtype=bool).flatten()
        if antithetic.size==1 and dim>1:
            antithetic = np.repeat(antithetic, dim)

        N = np.sum(1*np.array(antithetic))
        order_,order = order,int(order*2**-N+1*(order%2!=0))
        trans_ = trans
        trans = lambda X, verbose: \
                trans_(antithetic_gen(X, antithetic)[:,:order_])

    if rule=="C":
        X = chebyshev(dim, order)
    elif rule=="NC":
        X = chebyshev_nested(dim, order)
    elif rule=="G":
        X = _gw(order-1,dist)[0]
        trans = lambda x, verbose:x
    elif rule=="K":
        X = korobov(dim, order)
    elif rule=="R":
        X = np.random.random((dim,order))
    elif rule=="RG":
        X = regular_grid(dim, order)
    elif rule=="NG":
        X = regular_grid_nested(dim, order)
    elif rule=="L":
        X = latin_hypercube(dim, order)
    elif rule=="S":
        X = sobol(dim, order)
    elif rule=="H":
        X = halton(dim, order)
    elif rule=="M":
        X = hammersley(dim, order)
    else:
        raise KeyError("rule not recognised")

    X = trans(X, verbose=verbose)

    return X



def sample_select(sample, dist):
    """
Select collocation method from name

Parameters
----------
sample : str

    Format "<CGHSUYO><BM><IA><#><LED>" where
    <CGHSRUYO> : Defines sampling scheme
        C   Chebishev nodes (cosine transformed uniform)
        G   Gaussian quadrature
        H   Halton sampling (default)
        K   Korobov samples
        S   Stroud's cubature
        R   Pseudo-random samples
        U   Uniform nodes
        Y   Latin Hypercube Sampling
        O   Sobol sampling
    <BMN> : Defines how nodes are related to the distribution (except
            for Gaussian quadrature)
        B   Use a reasonable upper and lower bound and distribute
            the samples on the interval.
        M   Map samples using an inverse Rosenblatt-transform.
            (default)
        N   No mapping
    <IA> : Decides if the endpoints of the sample interval should
            be used (if any).
        I   Only interior points (default)
        A   Use all points
    <#> : int
        Order of sparse grid to use
        0   Classical full tensor grid (default)
    <LED> : Sample growth rule
        L   Linear rule N (default)
        E   Exponential rule 2**N (is nested for C and U nodes)
        D   Double the number of polynomial terms (recommended for
            H, R and Y)


Returns
-------
sample_func : callable(M)
    M : int
        Quadrature order
    """

    print("Warning: sample_select is depricated. Use samplegen instead")

    sample = sample.upper()

    if "G" in sample: scheme = 0
    elif "U" in sample: scheme = 1
    elif "C" in sample: scheme = 2
    elif "S" in sample: scheme = 3
    elif "Y" in sample: scheme = 4
    elif "R" in sample: scheme = 5
    elif "H" in sample: scheme = 6
    elif "O" in sample: scheme = 7
    elif "K" in sample: scheme = 8
    else:
        scheme = 7

    if "N" in sample: box = 2
    elif "M" in sample: box = 0
    elif "B" in sample: box = 1
    else:
        box = 0

    if "A" in sample: edge = True
    elif "I" in sample: edge = False
    else:
        edge = False

    if "L" in sample: growth = 0
    elif "E" in sample: growth = 1
    elif "D" in sample: growth = 2
    else:
        growth = 0

    i = 0
    for i in xrange(len(sample)):
        if sample[i].isdigit(): break
    j = i
    for j in xrange(i+1, len(sample)):
        if not sample[j].isdigit(): break
    if i!=j:
        sparse = int(sample[i:j])
    else:
        sparse = 0

    return Sampler(dist, scheme=scheme, box=box, edge=edge,
            sparse=sparse, growth=growth)





class Sampler(object):
    """
Sample generator

Examples
--------

Order 3 Gaussian quadrature
>>> S = cp.Sampler(cp.Normal(), scheme=0)
>>> print(S(3))
[-2.33441422 -0.74196378  0.74196378  2.33441422]

Full tensor grid with Clenshaw-Curtis nodes
>>> dist = cp.J(cp.Uniform(), cp.Uniform())
>>> S = cp.Sampler(dist, scheme=2, edge=1)
>>> print(S(1))
[[ 0.   0. ]
 [ 0.   0.5]
 [ 0.   1. ]
 [ 0.5  0. ]
 [ 0.5  0.5]
 [ 0.5  1. ]
 [ 1.   0. ]
 [ 1.   0.5]
 [ 1.   1. ]]

Smolyak sparse grid with Fejer nodes
>>> S = cp.Sampler(dist, scheme=2, edge=0, sparse=2)
>>> print(S(1))
[[ 0.5   0.5 ]
 [ 0.25  0.5 ]
 [ 0.75  0.5 ]
 [ 0.5   0.25]
 [ 0.5   0.75]]

Stroud's cubature of order 3
>>> S = cp.Sampler(dist, scheme=3)
>>> print(S(3))
[[ 0.90824829  0.5       ]
 [ 0.5         0.90824829]
 [ 0.09175171  0.5       ]]

Latin Hypercube sampling
>>> cp.seed(1000)
>>> S = cp.Sampler(dist, scheme=4)
>>> print(S(3))
[[ 0.6633974   0.46811863]
 [ 0.27875174  0.05308317]
 [ 0.98757072  0.51017741]
 [ 0.12054785  0.84929862]]
    """

    def __init__(self, dist, scheme=1, box=True, edge=False,
            growth=0, sparse=0):
        """
Parameters
----------
dist : Dist
    The domain where the samples are to be generated

Optional parameters

box : int
    0 : The samples are mapped onto the domain using an inverse
        Rosenblatt transform.
    1 : Will only use distribution bounds and evenly
        distribute samples inbetween.
    2 : Non transformation will be performed.
edge: bool
    True : Will include the bounds of the domain in the samples.
        If infinite domain, a resonable trunkation will be used.
growth : int
    0 : Linear growth rule. Minimizes the number of samples.
    1 : Exponential growth rule. Nested for schemes 1 and 2.
    2 : Double the number of polynomial terms. Recommended for
        schemes 4, 5 and 6.
sparse : int
    Defines the sparseness of the samples.
    sparse=len(dist) is equivalent to Smolyak sparse grid nodes.
    0 : A full tensor product nodes will be used
scheme : int
    0 : Gaussian quadrature will be used.
        box and edge argument will be ignored.
    1 : uniform distributed samples will be used.
    2 : The roots of Chebishev polynomials will be used
        (Clenshaw-Curtis and Fejer).
    3 : Stroud's cubature rule.
        Only order 2 and 3 valid,
        edge, growth and sparse are ignored.
    4 : Latin Hypercube sampling, no edge or sparse.
    5 : Classical random sampling, no edge or sparse.
    6 : Halton sampling, no edge or sparse
    7 : Hammersley sampling, no edge or sparse
    8 : Sobol sampling, no edge or sparse
    9 : Korobov samples, no edge or sparse
        """
        print("Warning: Sampler is depricated. Use samplegen instead")

        self.dist = dist
        self.scheme = scheme
        self.sparse = sparse

        # Gausian Quadrature
        if scheme==0:
            segment = lambda n: _gw(n,dist)[0]
            self.trans = lambda x:x

        else:

            if box==0:
                self.trans = lambda x: self.dist.inv(x.T).T
            elif box==1:
                lo, up = dist.range().reshape(2, len(dist))
                self.trans = lambda x: np.array(x)*(up-lo) + lo
            elif box==2:
                self.trans = lambda x: x

            if scheme==1:
                _segment = lambda n: np.arange(0, n+1)*1./n

            elif scheme==2:
                _segment = lambda n: \
                    .5*np.cos(np.arange(n,-1,-1)*np.pi/n) + .5

        if scheme in (1,2):
            if edge:

                if growth==0:
                    segment = lambda n: _segment(n+1)

                elif growth==1:
                    segment = lambda n: _segment(2**n)

                elif growth==2:
                    segment = lambda n: _segment(2*be.terms(n, len(dist)))

            else:

                if growth==0:
                    segment = lambda n: _segment(n+2)[1:-1]

                elif growth==1:
                    segment = lambda n: _segment(2**(n+1))[1:-1]

                elif growth==2:
                    segment = lambda n: _segment(2*be.terms(n, \
                                len(dist)))[1:-1]


        elif scheme==4:
            if growth==0:
                segment = lambda n: latin_hypercube(n+1, len(dist))
            elif growth==1:
                segment = lambda n: latin_hypercube(2**n, len(dist))
            elif growth==2:
                segment = lambda n: latin_hypercube(2*be.terms(n, len(dist)), \
                        len(dist))

        elif scheme==5:
            if growth==0:
                segment = lambda n: np.random.random((n+1, len(dist)))
            elif growth==1:
                segment = lambda n: np.random.random((2**n, len(dist)))
            elif growth==2:
                segment = lambda n: np.random.random((2*be.terms(n+1,
                    len(dist)), len(dist)))


        elif scheme==6:
            if growth==0:
                segment = lambda n: halton(n+1, len(dist))
            elif growth==1:
                segment = lambda n: halton(2**n, len(dist))
            elif growth==2:
                segment = lambda n: halton(2*be.terms(n, len(dist)), \
                        len(dist))

        elif scheme==7:
            if growth==0:
                segment = lambda n: hammersley(n+1, len(dist))
            elif growth==1:
                segment = lambda n: hammersley(2**n, len(dist))
            elif growth==2:
                segment = lambda n: hammersley(2*be.terms(n, len(dist)), \
                        len(dist))

        elif scheme==8:
            if growth==0:
                segment = lambda n: sobol(n+1, len(dist))
            elif growth==1:
                segment = lambda n: sobol(2**n, len(dist))
            elif growth==2:
                segment = lambda n: sobol(2*be.terms(n, len(dist)), \
                        len(dist))

        elif scheme==9:
            if growth==0:
                segment = lambda n: korobov(n+1, len(dist))
            elif growth==1:
                segment = lambda n: korobov(2**n, len(dist))
            elif growth==2:
                segment = lambda n: korobov(2*be.terms(n, len(dist)), \
                        len(dist))


        self.segment = segment


    def __call__(self, N):
        """
Sample generator

Parameters
----------
N : int
    Upper quadrature order

Returns
-------
samples : np.ndarray
    The quadrature nodes with `samples.shape=(D,K)` where
    `D=len(dist)` and `K` is the number of nodes.
        """
        dim = len(self.dist)

        if self.sparse==0 or self.scheme in (3,4,5,6):
            X = self.segment(N)
            if self.scheme in (1,2):
                X = combine((X,)*dim)
            out = self.trans(X)

        else:
            out = []
            for i in xrange(be.terms(N-self.sparse, dim),
                    be.terms(N, dim)):
                I = be.multi_index(i, dim)
                out.append(combine([self.segment(n) for n in I]))
            out = self.trans(np.concatenate(out, 0))

        return out.T




def antithetic_gen(U, order):

    order = np.array(order, dtype=bool).flatten()
    if order.size==1 and len(U)>1:
        order = np.repeat(order, len(U))

    U = np.asfarray(U).T
    iU = 1-U
    out = []
    index = np.zeros(len(U.T), dtype=bool)

    def expand(I):
        index[order] = I
        return index

    for I in np.ndindex((2,)*sum(order*1)):
        I = expand(I)
        out.append((U*I + iU*(True-I)).T)

    out = np.concatenate(out[::-1], 1)
    return out

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


def latin_hypercube(dim, n):
    """
Latin Hypercube sampling
    """
    R = np.random.random(n*dim).reshape((dim, n))
    for d in xrange(dim):
        perm = np.random.permutation(n)
        R[d] = (perm + R[d])/n
    return R

def corput(p, k):
    """
Van der Corput sampling
    """
    k += 1
    out = 0.
    p_base = p
    while k>0:
        a = k % p_base
        out += a*1./p
        k /= p_base
        p *= p_base
    return out


def primes(n):
    """
Generate primes using sieve of Eratosthenes
    """
    if n==2: return [2]
    elif n<2: return []
    s=range(3,n+1,2)
    mroot = n ** 0.5
    half=(n+1)/2-1
    i=0
    m=3
    while m <= mroot:
        if s[i]:
            j=(m*m-3)/2
            s[j]=0
            while j<half:
                s[j]=0
                j+=m
        i=i+1
        m=2*i+3
    return [2]+[x for x in s if x]

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

def halton(dim,n):
    "Halton sequence"
    p = []
    m = 10
    while len(p)<dim:
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
    for i in xrange(1,dim):
        z[i] = a*z[i-1] % (n+1)

    grid = np.mgrid[:dim,:n+1]
    Z = z[grid[0]]
    B = grid[1]+1
    P = B*Z/(n+1) % 1
    return P[:,:n]


if __name__=="__main__":
    import __init__ as cp
    import doctest
    doctest.testmod()
