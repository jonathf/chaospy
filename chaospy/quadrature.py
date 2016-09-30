"""
Methods for performing Gaussian Quadrature

Functions
---------
quad            Clenshaw-Curtis or Gaussian quadrature
golub_welsch    General tool for generating general quadrature
                nodes, weights
stieltjes       Tool for performing discretized and analytical
                Stieltjes procedure.
"""


import numpy as np
from scipy.misc import comb
from scipy.optimize import fminbound
from scipy.linalg import eig_banded

import chaospy as cp


def generate_quadrature(order, domain, acc=100, sparse=False, rule="C",
            composite=1, growth=None, part=None, **kws):
    """
Numerical quadrature node and weight generator

Parameters
----------
order : int
    The order of the quadrature.
domain : array_like, Dist
    If array is provided domain is the lower and upper bounds
    (lo,up). Invalid if gaussian is set.
    If Dist is provided, bounds and nodes are adapted to the
    distribution. This includes weighting the nodes in
    Clenshaw-Curtis quadrature.
acc : int
    If gaussian is set, but the Dist provieded in domain does not
    provide an analytical TTR, ac sets the approximation order for
    the descitized Stieltje's method.
sparse : bool
    If True used Smolyak's sparse grid instead of normal tensor
    product grid.
rule : str
    Quadrature rule

    Key                 Description
    "Gaussian", "G"     Optimal Gaussian quadrature from Golub-Welsch
                        Slow for high order.
    "Legendre", "E"     Gauss-Legendre quadrature
    "Clenshaw", "C"     Clenshaw-Curtis quadrature. Exponential growth rule is
                        used when sparse is True to make the rule nested.
    "Leja", J"          Leja quadrature. Linear growth rule is nested.
    "Genz", "Z"         Hermite Genz-Keizter 16 rule. Nested. Valid to order 8.
    "Patterson", "P"    Gauss-Patterson quadrature rule. Nested. Valid to order 8.

    If other is provided, Monte Carlo integration is assumed and
    arguemnt is passed to samplegen with order and domain.

composite : int, optional
    If provided, composite quadrature will be used. Value determines
    the number of domains along an axis.
    Ignored in the case gaussian=True.
growth : bool, optional
    If True sets the growth rule for the composite quadrature
    rule to exponential for Clenshaw-Curtis quadrature.
    Selected from context if omitted.
**kws : optional
    Extra keywords passed to samplegen.

See also
--------
samplegen   Sample generator
    """

    rule = rule.upper()
    if      rule == "GAUSSIAN": rule = "G"
    elif    rule == "LEGENDRE": rule = "E"
    elif    rule == "CLENSHAW": rule = "C"
    elif    rule == "LEJA":     rule = "J"
    elif    rule == "GENZ":     rule = "Z"
    elif    rule == "PATTERSON":rule = "P"

    if rule != "C" and growth and order:
        if isinstance(order, int):
            order = 2**order
        else:
            order = tuple([2**o for o in order])

    if rule == "G":

        assert isinstance(domain, cp.dist.Dist)

        if sparse:
            func = lambda m: golub_welsch(m, domain, acc)
            order = np.ones(len(domain), dtype=int)*order
            m = np.min(order)
            skew = [o-m for o in order]
            x, w = sparse_grid(func, m, len(domain), skew=skew)
        else:
            x, w = golub_welsch(order, domain, acc)

    elif rule in "ECJZP":

        isdist = not isinstance(domain, (tuple, list, np.ndarray))

        if isdist:
            lo,up = domain.range()
            dim = len(domain)
        else:
            lo,up = np.array(domain)
            dim = lo.size

        if sparse:
            if rule=="C":
                if growth is None:
                    growth = True
                func = lambda m: clenshaw_curtis(m, lo, up,
                        growth=growth, composite=composite)
            elif rule=="E":
                func = lambda m: gauss_legendre(m, lo, up,
                        composite)
            elif rule=="J":
                func = lambda m: leja(m, domain)
            elif rule=="Z":
                func = lambda m: cp.genz_keister.gk(m, domain)
            elif rule=="P":
                func = lambda m: cp.gauss_patterson.gp(m, domain)
            order = np.ones(dim, dtype=int)*order
            m = np.min(order)
            skew = [o-m for o in order]
            x, w = sparse_grid(func, m, dim, skew=skew)

        else:
            if rule=="C":
                if growth is None:
                    growth = False
                x, w = clenshaw_curtis(order, lo, up, growth=growth,
                        composite=composite)
                # foo = [lambda o: clenshaw_curtis(o[i], lo[i], up[i],
                #     growth=growth, composite=composite) for i in range(dim)]

            elif rule=="E":
                x, w = gauss_legendre(order, lo, up, composite)
                # foo = [lambda o: gauss_legendre(o[i], lo[i], up[i],
                #     composite) for i in range(dim)]

            elif rule=="J":
                x, w = leja(order, domain)

            elif rule=="Z":
                x, w = cp.genz_keister.gk(order, domain)

            if dim == 1:
                x = x.reshape(1, x.size)

        assert len(w) == x.shape[1]
        assert len(x.shape) == 2

        if isdist and not sparse:

            W = np.sum(w)

            eps = 1
            while (W-np.sum(w[w>eps]))>1e-18:
                eps *= .1

            valid = w>eps
            x, w = x[:, valid], w[valid]
            w /= np.sum(w)

    else:

        x = cp.dist.samplegen(order, domain, rule, **kws)
        w = np.ones(x.shape[-1])/x.shape[-1]

    return x, w


def golub_welsch(order, dist, acc=100, **kws):
    """
Golub-Welsch algorithm for creating quadrature nodes and weights

Parameters
----------
order : int
    Quadrature order
dist : Dist
    Distribution nodes and weights are found for with
    `dim=len(dist)`
acc : int
    Accuracy used in discretized Stieltjes procedure.
    Will be increased by one for each itteration.

Returns
-------
x : numpy.array
    Optimal collocation nodes with `x.shape=(dim, order+1)`
w : numpy.array
    Optimal collocation weights with `w.shape=(order+1,)`

Examples
--------
>>> Z = cp.Normal()
>>> x, w = cp.golub_welsch(3, Z)
>>> print(x)
[[-2.33441422 -0.74196378  0.74196378  2.33441422]]
>>> print(w)
[ 0.04587585  0.45412415  0.45412415  0.04587585]

Multivariate
>>> Z = cp.J(cp.Uniform(), cp.Uniform())
>>> x, w = cp. golub_welsch(1, Z)
>>> print(x)
[[ 0.21132487  0.21132487  0.78867513  0.78867513]
 [ 0.21132487  0.78867513  0.21132487  0.78867513]]
>>> print(w)
[ 0.25  0.25  0.25  0.25]
    """

    o = np.array(order)*np.ones(len(dist), dtype=int)+1
    P, g, a, b = stieltjes(dist, np.max(o), acc=acc, retall=True, **kws)

    X, W = [], []
    dim = len(dist)

    for d in range(dim):
        if o[d]:
            A = np.empty((2, o[d]))
            A[0] = a[d, :o[d]]
            A[1, :-1] = np.sqrt(b[d, 1:o[d]])
            vals, vecs = eig_banded(A, lower=True)

            x, w = vals.real, vecs[0, :]**2
            indices = np.argsort(x)
            x, w = x[indices], w[indices]

        else:
            x, w = np.array([a[d, 0]]), np.array([1.])

        X.append(x)
        W.append(w)

    if dim==1:
        x = np.array(X).reshape(1,o[0])
        w = np.array(W).reshape(o[0])
    else:
        x = cp.utils.combine(X).T
        w = np.prod(cp.utils.combine(W), -1)

    assert len(x)==dim
    assert len(w)==len(x.T)
    return x, w


def stieltjes(dist, order, acc=100, normed=False, retall=False,
        **kws):
    """
Discretized Stieltjes method

Parameters
----------
dist : Dist
    Distribution defining the space to create weights for.
order : int
    The polynomial order create.
acc : int
    The quadrature order of the Clenshaw-Curtis nodes to use at
    each step, if approximation is used.
retall : bool
    If included, more values are returned

Returns
-------
orth[, norms, A, B]

orth : list
    List of polynomials created from the method with
    `len(orth)==order+1`.
    If `len(dist)>1`, then each polynomials are multivariate.
norms : np.ndarray
    The norms of the polynomials with `norms.shape=(dim,order+1)`
    where `dim` are the number of dimensions in dist.
A,B : np.ndarray
    The three term coefficients. Both have `shape=(dim,order+1)`.

Examples
--------
>>> dist = cp.Uniform()
>>> orth, norms, A, B = cp.stieltjes(dist, 2, retall=True)
>>> print(cp.around(orth[2], 8))
[q0^2-q0+0.16666667]
>>> print(norms)
[[ 1.          0.08333333  0.00555556]]
    """

    assert not dist.dependent()

    try:
        dim = len(dist)
        K = np.arange(order+1).repeat(dim).reshape(order+1, dim).T
        A,B = dist.ttr(K)
        B[:,0] = 1.

        x = cp.poly.variable(dim)
        if normed:
            orth = [x**0*np.ones(dim), (x-A[:,0])/np.sqrt(B[:,1])]
            for n in range(1,order):
                orth.append((orth[-1]*(x-A[:,n]) - orth[-2]*np.sqrt(B[:,n]))/np.sqrt(B[:,n+1]))
            norms = np.ones(B.shape)
        else:
            orth = [x-x, x**0*np.ones(dim)]
            for n in range(order):
                orth.append(orth[-1]*(x-A[:,n]) - orth[-2]*B[:,n])
            orth = orth[1:]
            norms = np.cumprod(B, 1)

    except NotImplementedError:

        bnd = dist.range()
        kws["rule"] = kws.get("rule", "C")
        assert kws["rule"].upper()!="G"
        q,w = generate_quadrature(acc, bnd, **kws)
        w = w*dist.pdf(q)

        dim = len(dist)
        x = cp.poly.variable(dim)
        orth = [x*0, x**0]

        inner = np.sum(q*w, -1)
        norms = [np.ones(dim), np.ones(dim)]
        A,B = [],[]

        for n in range(order):

            A.append(inner/norms[-1])
            B.append(norms[-1]/norms[-2])
            orth.append((x-A[-1])*orth[-1] - orth[-2]*B[-1])

            y = orth[-1](*q)**2*w
            inner = np.sum(q*y, -1)
            norms.append(np.sum(y, -1))

            if normed:
                orth[-1] = orth[-1]/np.sqrt(norms[-1])

        A, B = np.array(A).T, np.array(B).T
        norms = np.array(norms[1:]).T
        orth = orth[1:]

    if retall:
        return orth, norms, A, B
    return orth


def weightgen(nodes, dist):
    poly = stieltjes(dist, len(nodes)-1, retall=True)[0]
    poly = cp.poly.flatten(cp.poly.Poly(poly))
    V = poly(nodes)
    Vi = np.linalg.inv(V)
    return Vi[:,0]

def leja(order, dist):
    """
After paper by Narayan and Jakeman
    """

    if len(dist) > 1:
        if isinstance(order, int):
            xw = [leja(order, d) for d in dist]
        else:
            xw = [leja(order[i], dist[i]) for i in range(len(dist))]

        x = [_[0][0] for _ in xw]
        w = [_[1] for _ in xw]
        x = cp.utils.combine(x).T
        w = cp.utils.combine(w)
        w = np.prod(w, -1)

        return x, w

    lo, up = dist.range()
    X = [lo, dist.mom(1), up]
    for o in range(order):

        X_ = np.array(X[1:-1])
        obj = lambda x:-np.sqrt(dist.pdf(x))*np.prod(np.abs(X_-x))
        opts, vals = zip(*[fminbound(obj, X[i], X[i+1],
            full_output=1)[:2] for i in range(len(X)-1)])
        index = np.argmin(vals)
        X.insert(index+1, opts[index])

    X = np.asfarray(X).flatten()[1:-1]
    W = weightgen(X, dist)
    X = X.reshape(1, X.size)

    return np.array(X), np.array(W)



def _clenshaw_curtis(N, composite=[]):

    if N==0:
        return np.array([.5]), np.array([1.])

    x = -np.cos(np.arange(N+1)*np.pi/N)
    x[np.abs(x)<1e-14] = 0

    k,n = np.meshgrid(*[np.arange(N/2+1)]*2)
    D = 2./N*np.cos(2*n*k*np.pi/N)
    D[:,0] *= .5
    D[:,-1] *= .5

    d = 2./(1-np.arange(0,N+1,2)**2)
    d[0] *= .5
    d[-1] *= .5

    w = np.dot(D.T, d)
    w = np.concatenate((w, w[-1-1*(N%2==0)::-1]))
    w[N/2] *= 2

    x = .5*x+.5
    w *= .5

    M = len(x)

    composite = list(set(composite))
    composite = [c for c in composite if (c<1) and (c>0)]
    composite.sort()
    composite = [0]+composite+[1]

    X = np.zeros((M-1)*(len(composite)-1)+1)
    W = np.zeros((M-1)*(len(composite)-1)+1)
    for d in range(len(composite)-1):
        X[d*M-d:(d+1)*M-d] = \
                x*(composite[d+1]-composite[d]) + composite[d]
        W[d*M-d:(d+1)*M-d] += w*(composite[d+1]-composite[d])

    return X, W


def clenshaw_curtis(N, lo=0, up=1, growth=False, composite=1, part=None):
    """
Generate the quadrature nodes and weights in Clenshaw-Curtis
quadrature
    """
    N,lo,up = [np.array(_).flatten() for _ in [N,lo,up]]
    dim = max(lo.size, up.size, N.size)
    N,lo,up = [np.ones(dim)*_ for _ in [N,lo,up]]
    N = np.array(N, dtype=int)

    if isinstance(composite, int):
        composite = [np.linspace(0,1,composite+1)]*dim
    else:
        composite = np.array(composite)
        if not composite.shape:
            composite = composite.flatten()
        if len(composite.shape)==1:
            composite = np.array([composite])
        composite = ((composite.T-lo)/(up-lo)).T

    if growth:
        q = [_clenshaw_curtis(2**N[i]-1*(N[i]==0), composite[i]) \
                for i in range(dim)]
    else:
        q = [_clenshaw_curtis(N[i], composite[i]) for i in range(dim)]

    x = [_[0] for _ in q]
    w = [_[1] for _ in q]

    x = cp.utils.combine(x, part=part).T
    w = cp.utils.combine(w, part=part)

    x = ((up-lo)*x.T + lo).T
    w = np.prod(w*(up-lo), -1)

    assert len(x)==dim
    assert len(w)==len(x.T)

    return x, w

def _gauss_legendre(N, composite=1):

    a = np.ones(N+1)*0.5
    b = np.arange(N+1)**2
    b = b/(16*b-4.)

    J = np.diag(np.sqrt(b[1:]), k=-1) + np.diag(a) + \
            np.diag(np.sqrt(b[1:]), k=1)
    vals, vecs = np.linalg.eig(J)

    x, w = vals.real, vecs[0,:]**2
    indices = np.argsort(x)
    x, w = x[indices], w[indices]

    M = len(x)

    composite = list(set(composite))
    composite = [c for c in composite if (c<1) and (c>0)]
    composite.sort()
    composite = [0]+composite+[1]

    X = np.empty(M*(len(composite)-1))
    W = np.empty(M*(len(composite)-1))
    for d in range(len(composite)-1):
        X[d*M:(d+1)*M] = \
                x*(composite[d+1]-composite[d]) + composite[d]
        W[d*M:(d+1)*M] = w*(composite[d+1]-composite[d])

    return X, W


def gauss_legendre(N, lo=0, up=1, composite=1):
    """
Generate the quadrature nodes and weights in Gauss-Legendre
quadrature
    """

    N,lo,up = [np.array(_).flatten() for _ in [N,lo,up]]
    dim = max(lo.size, up.size, N.size)
    N,lo,up = [np.ones(dim)*_ for _ in [N,lo,up]]
    N = np.array(N, dtype=int)

    if isinstance(composite, int):
        composite = [np.linspace(0,1,composite+1)]*dim
    else:
        composite = np.array(composite)
        if not composite.shape:
            composite = composite.flatten()
        if len(composite.shape)==1:
            composite = np.array([composite]).T
        composite = ((composite.T-lo)/(up-lo)).T


    q = [_gauss_legendre(N[i], composite[i]) for i in range(dim)]
    x = np.array([_[0] for _ in q])
    w = np.array([_[1] for _ in q])

    x = cp.utils.combine(x)
    w = cp.utils.combine(w)

    x = (up-lo)*x + lo
    w = np.prod(w*(up-lo), 1)

    return x.T, w


def sparse_grid(func, order, dim, skew=None):

    X, W = [], []
    bindex = cp.bertran.bindex(order-dim+1, order, dim)

    if skew is None:
        skew = np.zeros(dim, dtype=int)
    else:
        skew = np.array(skew, dtype=int)
        assert len(skew)==dim

    for i in range(cp.bertran.terms(
                order, dim) - cp.bertran.terms(order-dim, dim)):

        I = bindex[i]
        x,w = func(skew+I)
        w *= (-1)**(order-sum(I)) * comb(dim-1, order-sum(I))
        X.append(x)
        W.append(w)

    X = np.concatenate(X, 1)
    W = np.concatenate(W, 0)

    X = np.around(X, 15)
    order = np.lexsort(tuple(X))
    X = X.T[order].T
    W = W[order]

    # identify non-unique terms
    diff = np.diff(X.T, axis=0)
    ui = np.ones(len(X.T), bool)
    ui[1:] = (diff!=0).any(axis=1)

    # merge duplicate nodes
    N = len(W)
    i = 1
    while i<N:
        while i<N and ui[i]: i+=1
        j = i+1
        while j<N and not ui[j]: j+=1
        if j-i>1:
            W[i-1] = np.sum(W[i-1:j])
        i = j+1

    X = X[:,ui]
    W = W[ui]

    return X, W


def quad(func, order, domain, acc=100, sparse=False,
        rule="C", composite=1, retall=False, **kws):
    """
Numerical integration using Quadrature of problems on the form
::int dist.pdf(x) func(x) dx::

Weights and nodes are either calculated using Golub Welsch or
Clenshaw-Curtis methods.
In the former case the recurrence coefficients are calculated
analytical if available or else from discretized Stieltjes
procedure using Clenshaw-Curtis quadrature.

Parameters
----------
func : callable(x)
    function to take quadrature over
    x : np.ndarray
        where x.shape=M.shape return shape Q must be consistent
domain : Dist
    Distribution to take density function used as weighting
    function, where `dim=len(dist)`
order : int, array_like
    Quadrature order
    If array provided, all elements must be non-negative int
    If int provided, it is converted to a length dim array.
acc : int
    Accuracy of discretized Stieltjes procedure.
rule : str
    Quadrature rule

    Key     Description
    "G"     Optiomal Gaussian quadrature from Golub-Welsch
            Slow for high order and composite is ignored.
    "E"     Gauss-Legendre quadrature
    "C"     Clenshaw-Curtis quadrature. Exponential growth rule is
            used when sparse is True to make the rule nested.

    If other is provided, Monte Carlo integration is assumed and
    passed to samplegen with order and domain.

composite : int, optional
    If provided, composite quadrature will be used. Value determines
    the number of domains along an axis.
    Ignored in the case if gaussian=True.
retall : bool
    If True return nodes, weights and evaluvation in addition to
    answer.
**kws : optional
    Extra keyword passed to samplegen.

Returns
-------
I[, x, w, y]

I : np.ndarray
    Integration estimate
x : np.ndarray
    Quadrature node with `x.shape=(K,D)` where `D=len(dist)`.
w : np.ndarray
    Quadrature weights with `x.shape=(K,D)` where `D=len(dist)`.
y : np.ndarray
    Function evaluation of func in x

Examples
--------
>>> dist = cp.Gamma()
>>> func = lambda x: x**3-1
>>> q, x, w, y = cp.quad(func, 1, dist, retall=True, rule="G")
>>> print(q)
[ 5.]
>>> print(x)
[[ 0.58578644  3.41421356]]
    """
    x, w = generate_quadrature(
        order, domain, acc=acc, sparse=sparse, rule=rule, composite=composite)
    y = np.array([func(_) for _ in x.T])
    q = np.sum((y.T*w).T, 0)
    if retall:
        return q, x, w, y
    return q


def dep_golub_welsch(dist, order, acc=100):

    if not dist.dependent:
        return golub_welsch(dist, order, acc)
    raise NotImplementedError


#  def _quad_dependent(func, order, dist, acc=40, orth=None,
#      args=(), kws={}, veceval=False, retall=False):
#
#      dim = len(dist)
#      if isinstance(order, (int, float)):
#          order = [order]*dim
#      indices = cp.dist.sort(dist)
#      grid = np.mgrid[[slice(0,order[i]+1,1) for i in indices]]
#      X = np.empty([dim,]+[order[i]+1 for i in indices])
#      W = np.ones([order[i]+1 for i in indices])
#
#      def _dep_quad(I, order, dist, X, W, grid, vals):
#
#          dim = len(dist)
#          i = len(I)
#          j = indices[i]
#
#          Z = cp.dist.Subset(dist, j, vals)
#          x,w = Golub_Welsch(order[j], Z, acc)
#          W[I] *= w[grid[(i,)+I]]
#          X[(i,)+I] = x[grid[(i,)+I]]
#
#          if i==dim-1:
#              return X, W
#
#          for k in range(order[j]+1):
#              vals[j] = x[k]
#              X, W = _dep_quad(I+(k,), order, dist, X, W, grid, vals)
#
#          return X,W
#
#      X,W = _dep_quad((), order, dist, X, W, grid, [0.]*dim)
#      X = X[indices]
#      X = X.reshape(dim, X.size/dim)
#      W = W.flatten()
#
#      if veceval:
#          Y = np.array(func(X, *args, **kws)).T
#      else:
#          Y = np.array([func(_, *args, **kws) for _ in X.T]).T
#
#      if not (orth is None):
#          Q = orth(*X)
#          shape = Y.shape[:-1] + Q.shape
#          Y = Y.reshape(np.prod(Y.shape[:-1]), Y.shape[-1],)
#          Q = Q.reshape(np.prod(Q.shape[:-1]), Q.shape[-1],)
#
#          t1,t2 = np.mgrid[:len(Y), :len(Q)]
#          Y = (Y[t1]*Q[t2]).reshape(shape)
#
#      out = np.sum(Y*W, -1).T
#      if retall:
#          return out, X.T
#      return out


def rule_generator(*funcs):
    """
Constructor for creating quadrature generator

Parameters
----------
*funcs : callable
    One dimensional integration rule where
    for func in funcs:
        nodes, weights = func(order)
    order : int
        Order of integration rule
    nodes : array_like
        Where to evaluate.
    weights : array_like
        Weights corresponding to each node.

Returns
-------
mv_rule : callable
    Multidimensional integration rule
    nodes, weights = rule_gen(order)
    order : int, array_like
        Order of integration rule. If array_like, order along each
        axis.
    nodes : np.ndarray
        Where to evaluate with nodes.shape==(D,K), where
        D=len(funcs) and K is the number of points to evaluate.
    weights : np.ndarray
        Weights to go with the nodes with weights.shape=(K,).
    """

    dim = len(funcs)
    def tensprod_rule(N, part=None):

        N = N*np.ones(dim, int)
        q = [funcs[i](N[i]) \
                for i in range(dim)]

        x = [np.array(_[0]).flatten() for _ in q]
        x = cp.utils.combine(x, part=part).T

        w = [np.array(_[1]).flatten() for _ in q]
        w = np.prod(cp.utils.combine(w, part=part), -1)

        return x, w
    tensprod_rule = cp.utils.lazy_eval(tensprod_rule)

    def mv_rule(order, sparse=False, growth=None, part=None):
        """
Multidimensional integration rule

Parameters
----------
order : int, array_like
    Order of integration rule. If array_like, order along each
    axis.

Returns
-------
nodes, weights

nodes : np.ndarray
    Where to evaluate with nodes.shape==(D,K), where
    D=len(funcs) and K is the number of points to evaluate.
weights : np.ndarray
    Weights to go with the nodes with weights.shape=(K,).
        """

        if growth:
            def foo(N):
                N = N*np.ones(dim, int)
                return tensprod_rule([growth(n) for n in N], part=part)
        else:
            def foo(N):
                return tensprod_rule(N, part=part)

        if sparse:
            order = np.ones(dim, dtype=int)*order
            m = np.min(order)
            skew = [o-m for o in order]
            return sparse_grid(foo, m, dim, skew=skew)
        return foo(order)

    return mv_rule

def momgen(order, domain, acc=100, sparse=False, rule="C",
        composite=1, part=None, trans=lambda x:x, **kws):

    if isinstance(domain, cp.dist.Dist):
        dim = len(domain)
    else:
        dim = np.array(domain[0]).size

    x0 = trans(np.zeros(dim))
    if np.array(x0).shape==():
        func = trans
        trans = lambda x: [func(x)]

    if part is None:
        X,W = generate_quadrature(order, domain=domain, acc=acc, sparse=sparse,
                rule=rule, composite=composite, part=part, **kws)
        Y = np.array(trans(X))

        def _mom(k):
            out = np.sum(np.prod(Y.T**k, -1)*W, 0)
            return out
    else:

        isdist = not isinstance(domain, (tuple, list, np.ndarray))
        if isdist:
            lo,up = domain.range()
        else:
            lo,up = np.array(domain)

        X,W,Y = [], [], []
        for I in np.ndindex(*part):
            x,w = clenshaw_curtis(order, lo, up, part=(I, part))
            y = np.array(trans(x))
            if isdist:
                w *= domain.pdf(x).flatten()
            if np.any(w):
                X.append(x); W.append(w); Y.append(y)

        def _mom(k):
            out = 0.
            for i in range(len(X)):
                out += np.sum(np.prod(Y[i].T**k, -1)*W[i], 0)
            return out

    _mom = cp.utils.lazy_eval(_mom, tuple)

    def mom(K, **kws):
        out = np.array([_mom(k) for k in K.T])
        return out
    return mom

def ttrgen(order, domain, **kws):

    prm = kws
    prm["acc"] = order
    prm["retall"] = True
    class TTR:
        def __init__(self, prm):
            self.running = False
            self.prm = prm
        def __call__(self1, self2, K, **kws):
            if self1.running:
                raise NotImplementedError
            self1.running = True

            a,b = stieltjes(domain, np.max(K)+1, **self1.prm)[2:]
            out = np.ones((2,) + K.shape)
            i = 0
            for k in K.T:
                j = 0
                for k_ in k:
                    if k_:
                        out[:,j,i] = a[j,k_],b[j,k_]
                    j += 1
                i += 1

            self1.running = False
            return out
    return TTR(prm)


def probabilistic_collocation(order, dist, subset=.1):
    X, W = golub_welsch(order, dist)

    p = dist.pdf(X)

    alpha = np.random.random(len(W))
    alpha = p>alpha*subset*np.max(p)

    X = X.T[alpha].T
    W = W[alpha]
    return X, W
