"""
Collection of approximation methods

Global methods are used on a distribution as a wrapper. Local
function are used by the graph-module as part of calculations.

Functions
---------
    pdf         Probability density function (local)
    pdf_full    Probability density function (global)
    ppf         Inverse CDF (local)
    inv         Inverse CDF (global)
    mom         Raw statistical moments (global)
    find_interior_point     Find an interior point (global)
"""
import numpy as np
from chaospy.quadrature import generate_quadrature


def pdf(dist, x, G, eps=1.e-7, verbose=False,
        retall=False):
    """
Calculate the probability density function locally.

Parameters
----------
dist : Dist
    Distribution in question. May not be an advanced variable.
x : np.ndarray
    Location coordinates. Requires that x.shape=(len(dist), K).
G : Graph
    The chaospy state of the distribution calculations.
eps : float
    Acceptable error level for the approximations
retall : bool
    If True return Graph with the next calculation state with the
    approximation.

Returns
-------
out[, G]

out : np.ndarray
    Local probability density function with out.shape=x.shape.
    To calculate actual density function: np.prod(out, 0)
G : Graph
    The chaospy calculation state after approximation is complete.
    """

    x = np.asfarray(x)
    lo,up = np.min(x), np.max(x)
    mu = .5*(lo+up)
    eps = np.where(x<mu, eps, -eps)

    G.__call__ = G.fwd_call

    out = np.empty(x.shape)
    for d in range(len(dist)):
        x[d] += eps[d]
        out[d] = G.copy()(x.copy(), dist)[d]
        x[d] -= eps[d]

    out = np.abs((out-G(x.copy(), dist))/eps)

    G.__call__ = G.pdf_call

    if retall:
        return out, G
    return out

def pdf_full(dist, x, eps=1.e-7, verbose=False,
        retall=False):
    """
Calculate the probability density function globaly.

Parameters
----------
dist : Dist
    Distribution in question. May not be an advanced variable.
x : np.ndarray
    Location coordinates. Requires that x.shape=(len(dist), K).
eps : float
    Acceptable error level for the approximations
retall : bool
    If True return Graph with the next calculation state with the
    approximation.

Returns
-------
out[, G]

out : np.ndarray
    Global probability density function with out.shape=x.shape.
    To calculate actual density function: np.prod(out, 0)
G : Graph
    The chaospy calculation state after approximation is complete.
    """

    dim = len(dist)
    x = np.asfarray(x)

    shape = x.shape
    x = x.reshape(dim, x.size/dim)
    xdx = x.copy()
    y,G = dist.fwd(x,retall=True)
    lo,up = dist.range(x)
    mu = .5*(lo+up)
    out = np.empty(shape)
    eps = eps*np.ones(dim)

    for i in range(dim):
        eps_ = np.where(x[i]<mu[i], eps[i], -eps[i])

        xdx[i] += eps_
        out[i] = np.abs(dist.fwd(xdx)[i]-y[i])/eps[i]
        xdx[i] -= eps_

    if retall:
        return out, G
    return out

def inv(dist, q, maxiter=100, tol=1e-5, retall=False,
        verbose=False):
    """
Calculate the approximation of the point percentile function.

Parameters
----------
dist : Dist
    Distribution to estimate ppf.
q : np.ndarray
    Input values. All values must be on [0,1] and
    `q.shape==(dim,size)` where dim is the number of dimensions in
    dist and size is the number of values to calculate
    simultaneously.
maxiter : int
    The maximum number of iterations allowed before aborting
tol : float
    Tolerance parameter determining convergence.
retall : bool
    If true, return all.

Returns
-------
x, itrs, y

x : np.ndarray
    Distribution definition values.
itrs : int
    The number of iterations used before converging.
y : np.ndarray
    The model forward transformed value in x
    """

    dim = len(dist)
    size = q.size/dim
    q = q.reshape(dim, size)
    lo,up = dist.range(np.zeros((dim, size)))
    lo = lo*np.ones((dim,size))
    up = up*np.ones((dim,size))

    span = .5*(up-lo)
    too_much = np.any(dist.fwd(lo)>0, 0)
    while np.any(too_much):
        lo[:,too_much] -= span[:,too_much]
        too_much[too_much] = np.any(dist.fwd(lo)[:,too_much]>0, 0)

    too_little = np.any(dist.fwd(up)<1, 0)
    while np.any(too_little):
        up[:, too_little] += span[:, too_little]
        too_little[too_little] = np.any(dist.fwd(up)[:,too_little]<1, 0)

    # Initial values
    x = (up-lo)*q + lo
    flo, fup = -q, 1-q
    fx = tol*10*np.ones((dim,size))
    div = np.any((x<up)*(x>lo), 0)

    for iteration in range(1, maxiter+1):

        # eval function
        fx[:,div] = dist.fwd(x)[:,div]-q[:,div]

        # convergence test
        div[div] = np.any(np.abs(fx)>tol, 0)[div]
        if not np.any(div):
            break

        dfx = dist.pdf(x)[:,div]
        dfx = np.where(dfx==0, np.inf, dfx)

        # reduce boundaries
        lo_,up_ = dist.range(x)
        flo[:,div] = np.where(fx<=0, fx, flo)[:,div]
        lo[:,div] = np.where(fx<=0, x, lo)[:,div]
        lo = np.min([lo_, lo], 0)

        fup[:,div] = np.where(fx>=0, fx, fup)[:,div]
        up[:,div] = np.where(fx>=0, x, up)[:,div]
        up = np.max([up_, up], 0)

        # Newton increment
        xdx = x[:,div]-fx[:,div]/dfx

        # if new val on interior use Newton
        # else binary search
        x[:,div] = np.where((xdx<up[:,div])*(xdx>lo[:,div]),
                xdx, .5*(up+lo)[:,div])


    if retall:
        return x, iteration, dist.fwd(x)
    return x


def ppf(dist, q, G, maxiter=100, tol=1e-5, retall=False,
        verbose=False):
    """
Calculate the approximation of the point percentile function.

Parameters
----------
dist : Dist
    Distribution to estimate ppf.
q : np.ndarray
    Input values. All values must be on [0,1] and
    `q.shape==(dim,size)` where dim is the number of dimensions in
    dist and size is the number of values to calculate
    simultaneously.
maxiter : int
    The maximum number of iterations allowed before aborting
tol : float
    Tolerance parameter determining convergence.
retall : bool
    If true, return all.

Returns
-------
x, itrs

x : np.ndarray
    Distribution definition values.
itrs : int
    The number of iterations used before converging.
    """

    if not dist.advance:
        dist.prm, prm = G.K.build(), dist.prm
        out = inv(dist, q, maxiter, tol, retall, verbose)
        dist.prm = prm
        return out

    dim = len(dist)
    shape = q.shape
    size = q.size/dim
    q = q.reshape(dim, size)

    X = G.copy().run(size, "rnd")[1].node[dist]["key"]

    x = np.mean(X, -1)
    lo,up = np.min(X, -1), np.max(X, -1)
    lo = (lo*np.ones((size,dim))).T
    up = (up*np.ones((size,dim))).T

    # Initial values
    x = ((up.T-lo.T)*q.T + lo.T).T
    flo, fup = -q, 1-q
    fx = Fx = tol*10*np.ones((dim,size))
    dfx = 1.

    for iteration in range(1, maxiter+1):

        try:
            # eval function
            fx = G.copy().fwd_call(x, dist)
            success = (fx>=0)*(fx<=1)
            Fx = fx-q

            dfx = G.copy().pdf_call(x, dist)
            dfx = np.where(dfx==0, np.inf, dfx)

        except:
            success = np.zeros(size, dtype=bool)

        # convergence test
        if np.all(success) and np.all(np.abs(fx)<tol):
            break

        # reduce boundaries
        flo = np.where((Fx<0)*success, Fx, flo)
        lo = np.where((Fx<0)*success, x, lo)

        fup = np.where((Fx>0)*success, Fx, fup)
        up = np.where((Fx>0)*success, x, up)

        # Newton increment
        xdx = x-Fx/dfx

        # if new val on interior use Newton
        # else binary search
        x = np.where(success, xdx, .5*(up+lo))

    x = x.reshape(shape)

    if retall:
        return x, iteration, Fx
    return x


def mom(dist, K, retall=False, control_var=None,
        **kws):
    """
Approxmethod for estimation of raw statistical moments.

Parameters
----------
dist : Dist
    Distribution domain with dim=len(dist)
K : np.ndarray
    The exponents of the moments of interest with shape (dim,K).

Optional keywords

control_var : Dist
    If provided will be used as a control variable to try to reduce
    the error.
acc : int, optional
    The order of quadrature/MCI
sparse : bool
    If True used Smolyak's sparse grid instead of normal tensor
    product grid in numerical integration.
rule : str
    Quadrature rule
    Key     Description
    ----    -----------
    "G"     Optiomal Gaussian quadrature from Golub-Welsch
            Slow for high order and composit is ignored.
    "E"     Gauss-Legendre quadrature
    "C"     Clenshaw-Curtis quadrature. Exponential growth rule is
            used when sparse is True to make the rule nested.

    Monte Carlo Integration
    Key     Description
    ----    -----------
    "H"     Halton sequence
    "K"     Korobov set
    "L"     Latin hypercube sampling
    "M"     Hammersley sequence
    "R"     (Pseudo-)Random sampling
    "S"     Sobol sequence

composit : int, array_like optional
    If provided, composit quadrature will be used.
    Ignored in the case if gaussian=True.

    If int provided, determines number of even domain splits
    If array of ints, determines number of even domain splits along
        each axis
    If array of arrays/floats, determines location of splits

antithetic : array_like, optional
    List of bool. Represents the axes to mirror using antithetic
    variable during MCI.
    """

    dim = len(dist)
    shape = K.shape
    size = K.size/dim
    K = K.reshape(dim,size)

    if dim>1:
        shape = shape[1:]

    order = kws.pop("order", 40)
    X,W = generate_quadrature(order, dist, **kws)


    grid = np.mgrid[:len(X[0]),:size]
    X = X.T[grid[0]].T
    K = K.T[grid[1]].T
    out = np.prod(X**K, 0)*W

    if not (control_var is None):

        Y = control_var.ppf(dist.fwd(X))
        mu = control_var.mom(np.eye(len(control_var)))

        if mu.size==1 and dim>1:
            mu = mu.repeat(dim)

        for d in range(dim):
            alpha = np.cov(out, Y[d])[0,1]/np.var(Y[d])
            out -= alpha*(Y[d]-mu)

    out = np.sum(out, -1)

    return out


def find_interior_point(dist):
    """
Find interior point using the range-function

Parameters
----------
dist : Dist
    Distribution to find interior on.

Returns
-------
interior_point : np.ndarray
    shape=(len(dist),)
    """

    bnd = dist.range(np.zeros(len(dist)))
    x = .5*(bnd[1]-bnd[0])

    for i in range(10):
        bnd = dist.range(x)
        x_ = .5*(bnd[1]-bnd[0])
        if np.allclose(x,x_):
            break
        x = x_

    return x
