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
import numpy

import chaospy.quad


def pdf(dist, x, G, eps=1.e-7, verbose=False,
        retall=False):
    """
Calculate the probability density function locally.

Parameters
----------
dist : Dist
    Distribution in question. May not be an advanced variable.
x : numpy.ndarray
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

out : numpy.ndarray
    Local probability density function with out.shape=x.shape.
    To calculate actual density function: numpy.prod(out, 0)
G : Graph
    The chaospy calculation state after approximation is complete.
    """

    x = numpy.asfarray(x)
    lo,up = numpy.min(x), numpy.max(x)
    mu = .5*(lo+up)
    eps = numpy.where(x<mu, eps, -eps)

    G.__call__ = G.fwd_call

    out = numpy.empty(x.shape)
    for d in range(len(dist)):
        x[d] += eps[d]
        out[d] = G.copy()(x.copy(), dist)[d]
        x[d] -= eps[d]

    out = numpy.abs((out-G(x.copy(), dist))/eps)

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
x : numpy.ndarray
    Location coordinates. Requires that x.shape=(len(dist), K).
eps : float
    Acceptable error level for the approximations
retall : bool
    If True return Graph with the next calculation state with the
    approximation.

Returns
-------
out[, G]

out : numpy.ndarray
    Global probability density function with out.shape=x.shape.
    To calculate actual density function: numpy.prod(out, 0)
G : Graph
    The chaospy calculation state after approximation is complete.
    """

    dim = len(dist)
    x = numpy.asfarray(x)

    shape = x.shape
    x = x.reshape(dim, x.size/dim)
    xdx = x.copy()
    y,G = dist.fwd(x,retall=True)
    lo,up = dist.range(x)
    mu = .5*(lo+up)
    out = numpy.empty(shape)
    eps = eps*numpy.ones(dim)

    for i in range(dim):
        eps_ = numpy.where(x[i]<mu[i], eps[i], -eps[i])

        xdx[i] += eps_
        out[i] = numpy.abs(dist.fwd(xdx)[i]-y[i])/eps[i]
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
q : numpy.ndarray
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

x : numpy.ndarray
    Distribution definition values.
itrs : int
    The number of iterations used before converging.
y : numpy.ndarray
    The model forward transformed value in x
    """

    dim = len(dist)
    size = q.size/dim
    q = q.reshape(dim, size)
    lo,up = dist.range(numpy.zeros((dim, size)))
    lo = lo*numpy.ones((dim,size))
    up = up*numpy.ones((dim,size))

    span = .5*(up-lo)
    too_much = numpy.any(dist.fwd(lo)>0, 0)
    while numpy.any(too_much):
        lo[:,too_much] -= span[:,too_much]
        too_much[too_much] = numpy.any(dist.fwd(lo)[:,too_much]>0, 0)

    too_little = numpy.any(dist.fwd(up)<1, 0)
    while numpy.any(too_little):
        up[:, too_little] += span[:, too_little]
        too_little[too_little] = numpy.any(dist.fwd(up)[:,too_little]<1, 0)

    # Initial values
    x = (up-lo)*q + lo
    flo, fup = -q, 1-q
    fx = tol*10*numpy.ones((dim,size))
    div = numpy.any((x<up)*(x>lo), 0)

    for iteration in range(1, maxiter+1):

        # eval function
        fx[:,div] = dist.fwd(x)[:,div]-q[:,div]

        # convergence test
        div[div] = numpy.any(numpy.abs(fx)>tol, 0)[div]
        if not numpy.any(div):
            break

        dfx = dist.pdf(x)[:,div]
        dfx = numpy.where(dfx==0, numpy.inf, dfx)

        # reduce boundaries
        lo_,up_ = dist.range(x)
        flo[:,div] = numpy.where(fx<=0, fx, flo)[:,div]
        lo[:,div] = numpy.where(fx<=0, x, lo)[:,div]
        lo = numpy.min([lo_, lo], 0)

        fup[:,div] = numpy.where(fx>=0, fx, fup)[:,div]
        up[:,div] = numpy.where(fx>=0, x, up)[:,div]
        up = numpy.max([up_, up], 0)

        # Newton increment
        xdx = x[:,div]-fx[:,div]/dfx

        # if new val on interior use Newton
        # else binary search
        x[:,div] = numpy.where((xdx<up[:,div])*(xdx>lo[:,div]),
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
q : numpy.ndarray
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

x : numpy.ndarray
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

    x = numpy.mean(X, -1)
    lo,up = numpy.min(X, -1), numpy.max(X, -1)
    lo = (lo*numpy.ones((size,dim))).T
    up = (up*numpy.ones((size,dim))).T

    # Initial values
    x = ((up.T-lo.T)*q.T + lo.T).T
    flo, fup = -q, 1-q
    fx = Fx = tol*10*numpy.ones((dim,size))
    dfx = 1.

    for iteration in range(1, maxiter+1):

        try:
            # eval function
            fx = G.copy().fwd_call(x, dist)
            success = (fx>=0)*(fx<=1)
            Fx = fx-q

            dfx = G.copy().pdf_call(x, dist)
            dfx = numpy.where(dfx==0, numpy.inf, dfx)

        except:
            success = numpy.zeros(size, dtype=bool)

        # convergence test
        if numpy.all(success) and numpy.all(numpy.abs(fx)<tol):
            break

        # reduce boundaries
        flo = numpy.where((Fx<0)*success, Fx, flo)
        lo = numpy.where((Fx<0)*success, x, lo)

        fup = numpy.where((Fx>0)*success, Fx, fup)
        up = numpy.where((Fx>0)*success, x, up)

        # Newton increment
        xdx = x-Fx/dfx

        # if new val on interior use Newton
        # else binary search
        x = numpy.where(success, xdx, .5*(up+lo))

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
K : numpy.ndarray
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
    size = int(K.size/dim)
    K = K.reshape(dim,size)

    if dim>1:
        shape = shape[1:]

    order = kws.pop("order", 40)
    X,W = chaospy.quad.generate_quadrature(order, dist, **kws)


    grid = numpy.mgrid[:len(X[0]),:size]
    X = X.T[grid[0]].T
    K = K.T[grid[1]].T
    out = numpy.prod(X**K, 0)*W

    if not (control_var is None):

        Y = control_var.ppf(dist.fwd(X))
        mu = control_var.mom(numpy.eye(len(control_var)))

        if mu.size==1 and dim>1:
            mu = mu.repeat(dim)

        for d in range(dim):
            alpha = numpy.cov(out, Y[d])[0,1]/numpy.var(Y[d])
            out -= alpha*(Y[d]-mu)

    out = numpy.sum(out, -1)

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
interior_point : numpy.ndarray
    shape=(len(dist),)
    """
    try:
        x = dist.inv([.5]*len(dist))
        return x
    except:
        pass

    bnd = dist.range(numpy.zeros(len(dist)))
    x = .5*(bnd[1]-bnd[0])

    for i in range(10):
        bnd = dist.range(x)
        x_ = .5*(bnd[1]-bnd[0])
        if numpy.allclose(x, x_):
            break
        x = x_

    return x

# TODO: integrate these two functions.
def ttr(order, domain, **kws):

    prm = kws
    prm["accuracy"] = order
    prm["retall"] = True

    def _three_terms_recursion(self, keys, **kws):
        _, _, coeffs1, coeffs2 = chaospy.quad.generate_stieltjes(
            domain, numpy.max(keys)+1, **self1.prm)
        out = numpy.ones((2,) + keys.shape)
        idx = 0
        for idzs in keys.T:
            idy = 0
            for idz in idzs:
                if idz:
                    out[:, idy, idx] = coeffs1[idy, idz], coeffs2[idy, idz]
                idy += 1
            idx += 1

    return _three_terms_recursion

def moment_generator(order, domain, accuracy=100, sparse=False, rule="C",
                     composite=1, part=None, trans=lambda x:x, **kws):
    """Moment generator."""
    if isinstance(domain, chaospy.dist.Dist):
        dim = len(domain)
    else:
        dim = numpy.array(domain[0]).size

    if not numpy.array(trans(numpy.zeros(dim))).shape:
        func = trans
        trans = lambda x: [func(x)]

    if part is None:

        abscissas, weights = chaospy.quad.generate_quadrature(
            order, domain=domain, accuracy=accuracy, sparse=sparse,
            rule=rule, composite=composite, part=part, **kws)
        values = numpy.transpose(trans(abscissas))

        def moment_function(keys):
            """Raw statistical moment function."""
            return numpy.sum(numpy.prod(values**keys, -1)*weights, 0)
    else:

        isdist = isinstance(domain, chaospy.dist.Dist)
        if isdist:
            lower, upper = domain.range()
        else:
            lower, upper = numpy.array(domain)

        abscissas = []
        weights = []
        values = []
        for idx in numpy.ndindex(*part):
            abscissa, weight = chaospy.quad.collection.clenshaw_curtis(
                order, lower, upper, part=(idx, part))
            value = numpy.array(trans(abscissa))

            if isdist:
                weight *= domain.pdf(abscissa).flatten()

            if numpy.any(weight):
                abscissas.append(abscissa)
                weights.append(weight)
                values.append(value)

        def moment_function(keys):
            """Raw statistical moment function."""
            out = 0.
            for idx in range(len(abscissas)):
                out += numpy.sum(
                    numpy.prod(values[idx].T**keys, -1)*weights[idx], 0)
            return out

    def mom(keys, **kws):
        """Statistical moment function."""
        return numpy.array([moment_function(key) for key in keys.T])

    return mom
