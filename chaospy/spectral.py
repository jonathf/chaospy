r"""
In practice the following four components are needed to perform psuedo-spectral
projection. (For the "real" spectral projection method, see: :ref:`galerkin`):

-  A distribution for the unknown function parameters (as described in
   section :ref:`distributions`). For example::

      >>> distribution = cp.Iid(cp.Normal(0, 1), 2)

-  Create integration absissas and weights (as described in :ref:`quadrature`)::

      >>> absissas, weights = cp.generate_quadrature(
      ...     2, distribution, rule="G")
      >>> print(np.around(absissas, 15))
      [[-1.73205081 -1.73205081 -1.73205081  0.          0.          0.
         1.73205081  1.73205081  1.73205081]
       [-1.73205081  0.          1.73205081 -1.73205081  0.          1.73205081
        -1.73205081  0.          1.73205081]]
      >>> print(weights)
      [ 0.02777778  0.11111111  0.02777778  0.11111111  0.44444444  0.11111111
        0.02777778  0.11111111  0.02777778]

- An orthogonal polynomial expansion (as described in section
  :ref:`orthogonality`) where the weight function is the distribution in the
  first step::

      >>> orthogonal_expansion = cp.orth_ttr(2, distribution)
      >>> print(orthogonal_expansion)
      [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]

- A function evaluated using the nodes generated in the second step.
  For example::

      >>> def model_solver(q):
      ...     return [q[0]*q[1], q[0]*np.e**-q[1]+1]
      >>> solves = [model_solver(absissa) for absissa in absissas.T]
      >>> print(np.around(solves[:4], 8))
      [[ 3.         -8.7899559 ]
       [-0.         -0.73205081]
       [-3.          0.69356348]
       [-0.          1.        ]]

- To bring it together, expansion, absissas, weights and solves are used as
  arguments to create approximation::

      >>> approx = cp.fit_quadrature(
      ...     orthogonal_expansion, absissas, weights, solves)
      >>> print(cp.around(approx, 8))
      [q0q1, -1.58058656q0q1+1.63819248q0+1.0]

Note that in this case the function output is
bivariate. The software is designed to create an approximation of any
discretized model as long as it is compatible with ``numpy`` shapes.

As mentioned in section :ref:`orthogonality`, moment based construction of
polynomials can be unstable. This might also be the case for the
denominator :math:`\mathbb E{\Phi_n^2}`. So when using three terms
recursion, it is common to use the recurrence coefficients to estimated
the denominator.

One cavat with using psuedo-spectral projection is that the calculations of the
norms of the polynomials becomes unstable. To mittigate, recurrence
coefficients can be used to calculate them instead with more stability.
To include these stable norms in the calculations, the following change in code
can be added::

   >>> orthogonal_expansion, norms = cp.orth_ttr(2, distribution, retall=True)
   >>> approx2 = cp.fit_quadrature(
   ...     orthogonal_expansion, absissas, weights, solves, norms=norms)
   >>> print(cp.around(approx2, 8))
   [q0q1, -1.58058656q0q1+1.63819248q0+1.0]

Note that at low polynomial order, the error is very small. For example the
largest coefficient between the two approximation::

   >>> print(np.max(abs(approx-approx2).coeffs(), -1) < 1e-12)
   [ True  True]

The ``coeffs`` function returns all the polynomial coefficients.
"""

import numpy as np

try:
    from cubature._cubature import _cubature
except:
    pass

import chaospy as cp

__all__ = [
"pcm", "fit_adaptive", "fit_quadrature"
]


def pcm(func, porder, dist, rule="G", sorder=None, proxy_dist=None,
        orth=None, orth_acc=100, quad_acc=100, sparse=False, composit=1,
        antithetic=None, lr="LS", **kws):
    """
Probabilistic Collocation Method.

Args:
    func (callable) : The model to be approximated.  Must accept arguments on
            the form `func(z, *args, **kws)` where `z` is an 1-dimensional
            array with `len(z)==len(dist)`.
    porder (int) : The order of the polynomial approximation.
    dist (Dist) : Distributions for models parameter.
    rule (str) : The rule for estimating the Fourier coefficients.  For
            spectral projection/quadrature rules, see generate_quadrature.
            For point collocation/nummerical sampling, see samplegen.

Kwargs:
    proxy_dist (Dist) : If included, the expansion will be created in
            proxy_dist and values will be mapped to dist using a double
            Rosenblatt transformation.
    sorder (float) : The order of the sample scheme used.  If omited, default
            values will be used.
    orth (callable, Poly) : Orthogonal polynomial generation.
            If callable, the return of orth(order, dist) will be used. If Poly
            is provided, it will be used directly. If omited, orthogonal
            polynomial will be `orth_ttr` or `orth_chol` depending on if
            `dist` is stochastically independent of not.
    orth_acc (int) : Accuracy used in the estimation of polynomial expansion.
    sparse (bool) : If True, Smolyak sparsegrid will be used instead of full
            tensorgrid.
    composit (int) : Use composit rule. Note that the number of evaluations
            may grow quickly.
    antithetic (bool, array_like) : Use of antithetic variable
    lr (str) : Linear regresion method.
    lr_kws (dict) : Extra keyword arguments passed to fit_regression.

Returns:
    q : Poly
        Polynomial approximation of a given a model.

Examples:
    >>> func = lambda z: -z[1]**2 + 0.1*z[0]
    >>> dist = cp.J(cp.Uniform(), cp.Uniform())
    >>> q = cp.pcm(func, 2, dist)
    >>> print(cp.around(q, 10))
    -q1^2+0.1q0
    """
    # Proxy variable
    if proxy_dist is None:
        trans = lambda x:x
    else:
        dist, dist_ = proxy_dist, dist
        trans = lambda x: dist_.inv(dist.fwd(x), **kws)

    # The polynomial expansion
    if orth is None:
        if dist.dependent():
            orth = cp.orthogonal.orth_chol
        else:
            orth = cp.orthogonal.orth_ttr
    if not isinstance(orth, cp.poly.Poly):
        orth = orth(porder, dist, acc=orth_acc, **kws)

    # Applying scheme
    rule = rule.upper()
    if rule in "GEC":
        if sorder is None:
            sorder = porder+1
        z, w = cp.quadrature.generate_quadrature(
            sorder, dist, acc=quad_acc, sparse=sparse, rule=rule,
            composit=composit, **kws)

        x = trans(z)
        y = np.array([func(_) for _ in x.T])
        Q = fit_quadrature(orth, x, w, y, **kws)

    else:
        if sorder is None:
            sorder = 2*len(orth)
        z = dist.sample(sorder, rule=rule, antithetic=antithetic)

        x = trans(z)
        y = np.array(map(func, x.T))
        Q = fit_regression(orth, x, y, rule=lr, **kws)

    return Q


def pcm_cc(func, order, dist_out, dist_in=None, acc=None,
        orth=None, retall=False, sparse=False):
    """
Probabilistic Collocation Method using Clenshaw-Curtis quadrature

Parameters
----------
Required arguments

func : callable
    The model to be approximated.
    Must accept arguments on the form `func(z, *args, **kws)`
    where `z` is an 1-dimensional array with `len(z)==len(dist)`.
order : int
    The order of the polynomial approximation
dist_out : Dist
    Distributions for models parameter

Optional arguments

dist_in : Dist
    If included, space will be mapped using a Rosenblatt
    transformation from dist_out to dist_in before creating an
    expansin in terms of dist_in
acc : float
    The order of the sample scheme used
    If omitted order+1 will be used
orth : int, str, callable, Poly
    Orthogonal polynomial generation.

    int, str :
        orth will be passed to orth_select
        for selection of orthogonalization.
        See orth_select doc for more details.

    callable :
        the return of orth(M, dist) will be used.

    Poly :
        it will be used directly.
        All polynomials must be orthogonal for method to work
        properly.
retall : bool
    If True, return extra values.
sparse : bool
    If True, Smolyak sparsegrid will be used instead of full
    tensorgrid

Returns
-------
q[, x, w, y]

q : Poly
    Polynomial estimate of a given a model.
x : np.ndarray
    Nodes used in quadrature with `x.shape=(dim, K)` where K is the
    number of samples.
w : np.ndarray
    Weights used in quadrature with `w.shape=(K,)`.
y : np.ndarray
    Evauluations of func with `len(y)=K`.

#  Examples
#  --------
#
#  Define function and distribution:
#  >>> func = lambda z: -z[1]**2 + 0.1*z[0]
#  >>> dist = cp.J(cp.Uniform(), cp.Uniform())
#
#  Perform pcm:
#  >>> q, x, w, y = cp.pcm_cc(func, 2, dist, acc=2, retall=1)
#  >>> print(cp.around(q, 10))
#  -q1^2+0.1q0
#  >>> print(len(w))
#  9
#
#  With Smolyak sparsegrid
#  >>> q, x, w, y = cp.pcm_cc(func, 2, dist, acc=2, retall=1, sparse=1)
#  >>> print(cp.around(q, 10))
#  -q1^2+0.1q0
#  >>> print(len(w))
#  13
    """
    if acc is None:
        acc = order+1

    if dist_in is None:
        z, w = cp.quadrature.generate_quadrature(
            acc, dist_out, 100, sparse=sparse, rule="C")
        x = z
        dist = dist_out
    else:
        z, w = cp.quadrature.generate_quadrature(
            acc, dist_in, 100, sparse=sparse, rule="C")
        x = dist_out.ppf(dist_in.cdf(z))
        dist = dist_in

    if orth is None:
        if dist.dependent:
            orth = "chol"
        else:
            orth = "ttr"
    if isinstance(orth, (str, int)):
        orth = orth_select(orth)
    if not isinstance(orth, cp.poly.Poly):
        orth = orth(order, dist)

    y = np.array(map(func, x.T))
    Q = fit_quadrature(orth, x, w, y)

    if retall:
        return Q, x, w, y
    return Q

def fit_quadrature(orth, nodes, weights, solves, retall=False,
        norms=None, **kws):
    """
Using spectral projection to create a polynomial approximation over
distribution space.

Parameters
----------
orth : Poly
    Orthogonal polynomial expansion. Must be orthogonal for the
    approximation to be accurate.
nodes : array_like
    Where to evaluate the polynomial expansion and model to
    approximate.
    nodes.shape==(D,K) where D is the number of dimensions and K is
    the number of nodes.
weights : array_like
    Weights when doing numerical integration.
    weights.shape==(K,)
solves : array_like, callable
    The model to approximate.
    If array_like is provided, it must have len(solves)==K.
    If callable, it must take a single argument X with len(X)==D,
    and return a consistent numpy compatible shape.
norms : array_like
    In the of TTR using coefficients to estimate the polynomial
    norm is more stable than manual calculation.
    Calculated using quadrature if no provided.
    norms.shape==(len(orth),)
    """

    orth = cp.poly.Poly(orth)
    nodes = np.asfarray(nodes)
    weights = np.asfarray(weights)

    if hasattr(solves, "__call__"):
        solves = [solves(q) for q in nodes.T]
    solves = np.asfarray(solves)

    shape = solves.shape
    solves = solves.reshape(weights.size, solves.size/weights.size)

    ovals = orth(*nodes)
    vals1 = [(val*solves.T*weights).T for val in ovals]

    if norms is None:
        vals2 = [(val**2*weights).T for val in ovals]
        norms = np.sum(vals2, 1)
    else:
        norms = np.array(norms).flatten()
        assert len(norms)==len(orth)

    coefs = (np.sum(vals1, 1).T/norms).T
    coefs = coefs.reshape(len(coefs), *shape[1:])
    Q = cp.poly.transpose(cp.poly.sum(orth*coefs.T, -1))

    if retall:
        return Q, coefs
    return Q



def pcm_gq(func, order, dist_out, dist_in=None, acc=None,
        orth=None, retall=False, sparse=False):
    """
Probabilistic Collocation Method using optimal Gaussian quadrature

Parameters
----------
Required arguments

func : callable
    The model to be approximated.
    Must accept arguments on the form `func(z, *args, **kws)`
    where `z` is an 1-dimensional array with `len(z)==len(dist)`.
order : int
    The order of the polynomial approximation
dist_out : Dist
    Distributions for models parameter

Optional arguments

dist_in : Dist
    If included, space will be mapped using a Rosenblatt
    transformation from dist_out to dist_in before creating an
    expansin in terms of dist_in
acc : float
    The order of the sample scheme used
    If omitted order+1 will be used
orth : int, str, callable, Poly
    Orthogonal polynomial generation.

    int, str :
        orth will be passed to orth_select
        for selection of orthogonalization.
        See orth_select doc for more details.

    callable :
        the return of orth(M, dist) will be used.

    Poly :
        it will be used directly.
        All polynomials must be orthogonal for method to work
        properly.
args : itterable
    Extra positional arguments passed to `func`.
kws : dict
    Extra keyword arguments passed to `func`.
retall : bool
    If True, return also number of evaluations
sparse : bool
    If True, Smolyak sparsegrid will be used instead of full
    tensorgrid

Returns
-------
Q[, X]

Q : Poly
    Polynomial estimate of a given a model.
X : np.ndarray
    Values used in evaluation

#  Examples
#  --------
#  Define function:
#  >>> func = lambda z: z[1]*z[0]
#
#  Define distribution:
#  >>> dist = cp.J(cp.Normal(), cp.Normal())
#
#  Perform pcm:
#  >>> p, x, w, y = cp.pcm_gq(func, 2, dist, acc=3, retall=True)
#  >>> print(cp.around(p, 10))
#  q0q1
#  >>> print(len(w))
#  16

    """
    if acc is None:
        acc = order+1

    if dist_in is None:
        z, w = cp.quadrature.generate_quadrature(
            acc, dist_out, 100, sparse=sparse, rule="G")
        x = z
        dist = dist_out
    else:
        z, w = cp.quadrature.generate_quadrature(
            acc, dist_in, 100, sparse=sparse, rule="G")
        x = dist_out.ppf(dist_in.cdf(z))
        dist = dist_in

    y = np.array(map(func, x.T))
    shape = y.shape
    y = y.reshape(w.size, y.size/w.size)

    if orth is None:
        if dist.dependent:
            orth = "chol"
        else:
            orth = "ttr"
    if isinstance(orth, (str, int)):
        orth = orth_select(orth)
    if not isinstance(orth, cp.poly.Poly):
        orth = orth(order, dist)

    ovals = orth(*z)
    vals1 = [(val*y.T*w).T for val in ovals]
    vals2 = [(val**2*w).T for val in ovals]
    coef = (np.sum(vals1, 1).T/np.sum(vals2, 1)).T

    coef = coef.reshape(len(coef), *shape[1:])
    Q = cp.poly.transpose(cp.poly.sum(orth*coef.T, -1))

    if retall:
        return Q, x, w, y
    return Q


def pcm_lr(func, order, dist_out, sample=None,
        dist_in=None, rule="H",
        orth=3, regression="LS", retall=False):
    """
Probabilistic Collocation Method using Linear Least Squares fit

Parameters
----------
Required arguemnts

func : callable
    The model to be approximated.  Must accept arguments on the
    form `z` is an 1-dimensional array with `len(z)==len(dist)`.
order : int
    The order of chaos expansion.
dist_out : Dist
    Distributions for models parameter.

Optional arguments

sample : int
    The order of the sample scheme to be used.
    If omited it defaults to 2*len(orth).
dist_in : Dist
    If included, space will be mapped using a Rosenblatt
    transformation from dist_out to dist_in before creating an
    expansin in terms of dist_in
rule:
    rule for generating samples, where d is the number of
    dimensions.

    Key     Name                Nested
    ----    ----------------    ------
    "K"     Korobov             no
    "R"     (Pseudo-)Random     no
    "L"     Latin hypercube     no
    "S"     Sobol               yes
    "H"     Halton              yes
    "M"     Hammersley          yes

orth : int, str, callable, Poly
    Orthogonal polynomial generation.

    int, str :
        orth will be passed to orth_select
        for selection of orthogonalization.
        See orth_select doc for more details.

    callable :
        the return of orth(M, dist) will be used.

    Poly :
        it will be used directly.
        It must be of length N+1=comb(M+D, M)
regression : str
    Linear regression method used.
    See fit_regression for more details.
retall : bool
    If True, return extra values.

#  Examples
#  --------
#
#  Define function:
#  >>> func = lambda z: -z[1]**2 + 0.1*z[0]
#
#  Define distribution:
#  >>> dist = cp.J(cp.Normal(), cp.Normal())
#
#  Perform pcm:
#  >>> q, x, y = cp.pcm_lr(func, 2, dist, retall=True)
#  >>> print(cp.around(q, 10))
#  -q1^2+0.1q0
#  >>> print(len(x.T))
#  12
    """

    if dist_in is None:
        dist = dist_out
    else:
        dist = dist_in

    # orthogonalization
    if orth is None:
        if dist.dependent():
            orth = "chol"
        else:
            orth = "ttr"
    if isinstance(orth, (str, int)):
        orth = orth_select(orth)
    if not isinstance(orth, cp.poly.Poly):
        orth = orth(order, dist)

    # sampling
    if sample is None:
        sample = 2*len(orth)

    x = cp.dist.samplegen(sample, dist, rule)


    # Rosenblatt
    if not (dist_in is None):
        x = dist_out.ppf(dist_in.cdf(x))

    # evals
    y = np.array(map(func, x.T))
    shape = y.shape[1:]
    y = y.reshape(len(y), y.size/len(y))
    if sample==0:
        y_ = y[:]
        R = orth * y
    else:
        R, y_ = fit_regression(orth, x, y, regression, retall=1)

    R = cp.poly.reshape(R, shape)

    if retall:
        return R, x, y
    return R




def fit_lagrange(X, Y):
    """Simple lagrange method"""
    X = np.array(X)
    Y = np.array(Y)
    assert X.shape[0] == Y.shape[0]

    if len(X.shape) == 1:
        X = X.reshape(1, X.size)

    N, dim = X.shape

    basis = []
    n = 1
    while len(basis) < N:
        basis = cp.poly.basis(0, n, dim)
        n += 1

    basis = basis[:N]

    return fit_regression(basis, X, Y)



def cross_validate(P, X, Y, folds=None, rule="LS", **kws):
    """
Parameters
----------
P : Poly
    Polynomial expansion
X : array_like
    Input data with X.shape=(dim,K)
Y : array_like
    Output data with len(Y)=K
folds : int,optional
    Number of folds in validation. If omitted selected to be K.
    """
    X = np.array(X)
    dim,K = X.shape
    Y = np.array(Y)
    assert len(Y)==K

    out = Y.copy()

    if folds is None:
        folds = K
    R = np.random.randint(0, folds, K)

    for fold in range(folds):

        infold = R==fold
        x = X[:, True-infold]
        y = Y[True-infold]
        poly = fit_regression(P, x, y, rule=rule, **kws)
        out[infold] = out[infold]-poly(*X[:,infold])

    return out



def fit_adaptive(func, poly, dist, abserr=1.e-8, relerr=1.e-8,
        budget=0, norm=0, bufname="", retall=False):
    """Adaptive estimation of Fourier coefficients.

Parameters
----------
func : callable
    Should take a single argument `q` which is 1D array
    `len(q)=len(dist)`.
    Must return something compatible with np.ndarray.
poly : Poly
    Polynomial vector for which to create Fourier coefficients for.
dist : Dist
    A distribution to optimize the Fourier coefficients to.
abserr : float
    Absolute error tolerance.
relerr : float
    Relative error tolerance.
budget : int
    Soft maximum number of function evaluations.
    0 means unlimited.
norm : int
    Specifies the norm that is used to measure the error and
    determine convergence properties (irrelevant for single-valued
    functions). The `norm` argument takes one of the values:
    0 : L0-norm
    1 : L0-norm on top of paired the L2-norm. Good for complex
        numbers where each conseqtive pair of the solution is real
        and imaginery.
    2 : L2-norm
    3 : L1-norm
    4 : L_infinity-norm
bufname : str, optional
    Buffer evaluations to file such that the fit_adaptive can be
    run again without redooing all evaluations.
retall : bool
    If true, returns extra values.

Returns
-------
estimate[, coeffs, norms, coeff_error, norm_error]

estimate : Poly
    The polynomial chaos expansion representation of func.
coeffs : np.ndarray
    The Fourier coefficients.
norms : np.ndarray
    The norm of the orthogonal polynomial squared.
coeff_error : np.ndarray
    Estimated integration error of the coeffs.
norm_error : np.ndarray
    Estimated integration error of the norms.

Examples
--------
# >>> func = lambda q: q[0]*q[1]
# >>> poly = cp.basis(0,2,2)
# >>> dist = cp.J(cp.Uniform(0,1), cp.Uniform(0,1))
# >>> res = cp.fit_adaptive(func, poly, dist, budget=100)
# >>> print(res)
    """

    if bufname:
        func = cp.utils.lazy_eval(func, load=bufname)

    dim = len(dist)
    n = [0,0]

    dummy_x = dist.inv(.5*np.ones(dim, dtype=np.float64))
    val = np.array(func(dummy_x), np.float64)

    xmin = np.zeros(dim, np.float64)
    xmax = np.ones(dim, np.float64)

    def f1(u, ns, *args):
        qs = dist.inv(u.reshape(ns, dim))
        out = (poly(*qs.T)**2).T.flatten()
        return out
    dim1 = len(poly)
    val1 = np.empty(dim1, dtype=np.float64)
    err1 = np.empty(dim1, dtype=np.float64)
    _cubature(f1, dim1, xmin, xmax, (), "h", abserr, relerr, norm,
            budget, True, val1, err1)
    val1 = np.tile(val1, val.size)

    dim2 = np.prod(val.shape)*dim1
    val2 = np.empty(dim2, dtype=np.float64)
    err2 = np.empty(dim2, dtype=np.float64)
    def f2(u, ns, *args):
        n[0] += ns
        n[1] += 1
        qs = dist.inv(u.reshape(ns, dim))
        Y = np.array([func(q) for q in qs])
        Q = poly(*qs.T)
        out = np.array([Y.T*q1 for q1 in Q]).T.flatten()
        out = out/np.tile(val1, ns)
        return out
    try:
        _ = _cubature
    except:
        raise NotImplementedError(
                "cubature not install properly")
    _cubature(f2, dim2, xmin, xmax, (), "h", abserr, relerr, norm,
                budget, True, val2, err2)

    shape = (dim1,)+val.shape
    val2 = val2.reshape(shape[::-1]).T

    out = cp.poly.transpose(cp.poly.sum(poly*val2.T, -1))

    if retall:
        return out, val2, val1, err2, err1
    return val2

