"""
Tools for performing Probabilistic Collocation Method (PCM)

pcm             Front-end function for PCM
fit_adaptive    Fit an adaptive spectral projection
fit_regression  Fit a point collocation together
fit_quadrature  Fit a spectral projection together
lstsq_cv        Cross-validated least squares solver
rlstsq          Robust least squares solver
"""

import numpy as np
from scipy import linalg as la
from scipy import optimize as op

try:
    from sklearn import linear_model as lm
except:
    pass

try:
    from cubature._cubature import _cubature
except:
    pass

import poly as po
import quadrature as qu
import orthogonal
from dist import samplegen
from utils import lazy_eval

__version__ = "1.0"

__all__ = [
"pcm", "fit_adaptive", "fit_regression", "fit_quadrature", "lstsq_cv", "rlstsq"
]

def pcm(func, porder, dist, rule="G", sorder=None, proxy_dist=None,
        orth=None, orth_acc=100, quad_acc=100, sparse=False, composit=1,
        antithetic=None, lr="LS", **kws):
    """
Probabilistic Collocation Method

Parameters
----------
Required arguments

func : callable
    The model to be approximated.
    Must accept arguments on the form `func(z, *args, **kws)`
    where `z` is an 1-dimensional array with `len(z)==len(dist)`.
porder : int
    The order of the polynomial approximation
dist_out : Dist
    Distributions for models parameter
rule : str
    The rule for estimating the Fourier coefficients.
    For spectral projection/quadrature rules, see generate_quadrature.
    For point collocation/nummerical sampling, see samplegen.

Optional arguments

proxy_dist : Dist
    If included, the expansion will be created in proxy_dist and
    values will be mapped to dist using a double Rosenblatt
    transformation.
sorder : float
    The order of the sample scheme used.
    If omited, default values will be used.
orth : int, str, callable, Poly
    Orthogonal polynomial generation.

    int, str :
        orth will be passed to orth_select
        for selection of orthogonalization.
        See orth_select doc for more details.

    callable :
        the return of orth(order, dist) will be used.

    Poly :
        it will be used directly.
        All polynomials must be orthogonal for method to work
        properly if spectral projection is used.
orth_acc : int
    Accuracy used in the estimation of polynomial expansion.

Spectral projection arguments

sparse : bool
    If True, Smolyak sparsegrid will be used instead of full
    tensorgrid.
composit : int
    Use composit rule. Note that the number of evaluations may grow
    quickly.

Point collocation arguments

antithetic : bool, array_like
    Use of antithetic variable
lr : str
    Linear regresion method.
    See fit_regression for more details.
lr_kws : dict
    Extra keyword arguments passed to fit_regression.

Returns
-------
q : Poly
    Polynomial approximation of a given a model.

Examples
--------

Define function and distribution:
>>> func = lambda z: -z[1]**2 + 0.1*z[0]
>>> dist = cp.J(cp.Uniform(), cp.Uniform())

Perform pcm:
>>> q = cp.pcm(func, 2, dist)
>>> print cp.around(q, 10)
0.1q0-q1^2

See also
--------
generate_quadrature         Generator for quadrature rules
samplegen       Generator for sampling schemes
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
            orth = "svd"
        else:
            orth = "ttr"
    if isinstance(orth, (str, int, long)):
        orth = orthogonal.orth_select(orth, **kws)
    if not isinstance(orth, po.Poly):
        orth = orth(porder, dist, acc=orth_acc, **kws)

    # Applying scheme
    rule = rule.upper()
    if rule in "GEC":
        if sorder is None:
            sorder = porder+1
        z,w = qu.generate_quadrature(sorder, dist, acc=quad_acc, sparse=sparse,
                rule=rule, composit=composit, **kws)

        x = trans(z)
        y = np.array(map(func, x.T))
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
#  >>> print cp.around(q, 10)
#  -q1^2+0.1q0
#  >>> print len(w)
#  9
#  
#  With Smolyak sparsegrid
#  >>> q, x, w, y = cp.pcm_cc(func, 2, dist, acc=2, retall=1, sparse=1)
#  >>> print cp.around(q, 10)
#  -q1^2+0.1q0
#  >>> print len(w)
#  13
    """
    if acc is None:
        acc = order+1

    if dist_in is None:
        z,w = qu.generate_quadrature(acc, dist_out, 100, sparse=sparse,
                rule="C")
        x = z
        dist = dist_out
    else:
        z,w = qu.generate_quadrature(acc, dist_in, 100, sparse=sparse,
                rule="C")
        x = dist_out.ppf(dist_in.cdf(z))
        dist = dist_in

    if orth is None:
        if dist.dependent:
            orth = "chol"
        else:
            orth = "ttr"
    if isinstance(orth, (str, int, long)):
        orth = orth_select(orth)
    if not isinstance(orth, po.Poly):
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

    orth = po.Poly(orth)
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
    Q = po.transpose(po.sum(orth*coefs.T, -1))

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
#  >>> print cp.around(p, 10)
#  q0q1
#  >>> print len(w)
#  16

    """
    if acc is None:
        acc = order+1

    if dist_in is None:
        z,w = qu.generate_quadrature(acc, dist_out, 100, sparse=sparse,
                rule="G")
        x = z
        dist = dist_out
    else:
        z,w = qu.generate_quadrature(acc, dist_in, 100, sparse=sparse,
                rule="G")
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
    if isinstance(orth, (str, int, long)):
        orth = orth_select(orth)
    if not isinstance(orth, po.Poly):
        orth = orth(order, dist)

    ovals = orth(*z)
    vals1 = [(val*y.T*w).T for val in ovals]
    vals2 = [(val**2*w).T for val in ovals]
    coef = (np.sum(vals1, 1).T/np.sum(vals2, 1)).T

    coef = coef.reshape(len(coef), *shape[1:])
    Q = po.transpose(po.sum(orth*coef.T, -1))

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
#  >>> print cp.around(q, 10)
#  -q1^2+0.1q0
#  >>> print len(x.T)
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
    if isinstance(orth, (str, int, long)):
        orth = orth_select(orth)
    if not isinstance(orth, po.Poly):
        orth = orth(order, dist)

    # sampling
    if sample is None:
        sample = 2*len(orth)

    x = samplegen(sample, dist, rule)


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

    R = po.reshape(R, shape)

    if retall:
        return R, x, y
    return R

def lstsq_cv(A, b, order=1):
    A = np.array(A)
    b = np.array(b)
    m,l = A.shape

    if order==0:
        L = np.eye(l)
    elif order==1:
        L = np.zeros((l-1,l))
        L[:,:-1] -= np.eye(l-1)
        L[:,1:] += np.eye(l-1)
    elif order==2:
        L = np.zeros((l-2,l))
        L[:,:-2] += np.eye(l-2)
        L[:,1:-1] -= 2*np.eye(l-2)
        L[:,2:] += np.eye(l-2)
    elif order is None:
        L = np.zeros(1)
    else:
        L = np.array(order)
        assert L.shape[-1]==l or L.shape in ((), (1,))

#      def cross(alpha):
#          out = 0.
#          for k in range(l):
#              valid = np.arange(l)==k
#              A_ = A[:,valid]
#              b_ = b[valid]
#              _ = la.inv(np.dot(A_.T,A_) + alpha*np.dot(L.T, L))
#              out += np.dot(_, np.dot(A_.T, b_))

    return la.lstsq(A, b)



def rlstsq(A, b, order=1, alpha=None, cross=False, retall=False):
    """
Least Squares Minimization using Tikhonov regularization, and
robust generalized cross-validation.

Parameters
----------
A : array_like, shape (M,N)
    "Coefficient" matrix.
b : array_like, shape (M,) or (M, K)
    Ordinate or "dependent variable" values. If `b` is
    two-dimensional, the least-squares solution is calculated for
    each of the `K` columns of `b`.
order : int, array_like
    If int, it is the order of Tikhonov regularization.
    If array_like, it will be used as regularization matrix.
alpha : float, optional
    Lower threshold for the dampening parameter.
    The real value is calculated using generalised cross
    validation.
cross : bool
    Use cross validation
retall : bool
    If True, return also estimated alpha-value
    """

    A = np.array(A)
    b = np.array(b)
    m,l = A.shape

    if cross:
        out = np.empty((m,l) + b.shape[1:])
        A_ = np.empty((m-1,l))
        b_ = np.empty((m-1,) + b.shape[1:])
        for i in xrange(m):
            A_[:i] = A[:i]
            A_[i:] = A[i+1:]
            b_[:i] = b[:i]
            b_[i:] = b[i+1:]
            out[i] = rlstsq(A_, b_, order, alpha, False)

        return np.median(out, 0)

    if order==0:
        L = np.eye(l)

    elif order==1:
        L = np.zeros((l-1,l))
        L[:,:-1] -= np.eye(l-1)
        L[:,1:] += np.eye(l-1)

    elif order==2:
        L = np.zeros((l-2,l))
        L[:,:-2] += np.eye(l-2)
        L[:,1:-1] -= 2*np.eye(l-2)
        L[:,2:] += np.eye(l-2)

    elif order is None:
        L = np.zeros(1)

    else:
        L = np.array(order)
        assert L.shape[-1]==l or L.shape in ((), (1,))

    if alpha is None and not (order is None):

        gamma = 0.1

        def rgcv_error(alpha):
            if alpha<=0: return np.inf
            A_ = np.dot(A.T,A)+alpha*(np.dot(L.T,L))
            try:
                A_ = np.dot(la.inv(A_), A.T)
            except la.LinAlgError:
                return np.inf
            x = np.dot(A_, b)
            res2 = np.sum((np.dot(A,x)-b)**2)
            K = np.dot(A, A_)
            V = m*res2/np.trace(np.eye(m)-K)**2
            mu2 = np.sum(K*K.T)/m

            return (gamma + (1-gamma)*mu2)*V

        alpha = op.fmin(rgcv_error, 1, disp=0)

    out = la.inv(np.dot(A.T,A) + alpha*np.dot(L.T, L))
    out = np.dot(out, np.dot(A.T, b))
    if retall:
        return out, alpha
    return out


def fit_regression(P, x, u, rule="LS", retall=False, **kws):
    """
Fit a polynomial chaos expansion using linear regression.

Parameters
----------
P : Poly
    Polynomial chaos expansion with `P.shape=(M,)` and `P.dim=D`.
x : array_like
    Collocation nodes with `x.shape=(D,K)`.
u : array_like
    Model evaluations with `len(u)=K`.
retall : bool
    If True return uhat in addition to R
rule : str
    Regression method used.

    The follwong methods uses scikits-learn as backend.
    See `sklearn.linear_model` for more details.

    Key     Scikit-learn    Description
    ---     ------------    -----------
        Parameters      Description
        ----------      -----------

    "BARD"  ARDRegression   Bayesian ARD Regression
        n_iter=300      Maximum iterations
        tol=1e-3        Optimization tolerance
        alpha_1=1e-6    Gamma scale parameter
        alpha_2=1e-6    Gamma inverse scale parameter
        lambda_1=1e-6   Gamma shape parameter
        lambda_2=1e-6   Gamma inverse scale parameter
        threshold_lambda=1e-4   Upper pruning threshold

    "BR"    BayesianRidge   Bayesian Ridge Regression
        n_iter=300      Maximum iterations
        tol=1e-3        Optimization tolerance
        alpha_1=1e-6    Gamma scale parameter
        alpha_2=1e-6    Gamma inverse scale parameter
        lambda_1=1e-6   Gamma shape parameter
        lambda_2=1e-6   Gamma inverse scale parameter

    "EN"    ElastiNet       Elastic Net
        alpha=1.0       Dampening parameter
        rho             Mixing parameter in [0,1]
        max_iter=300    Maximum iterations
        tol             Optimization tolerance

    "ENC"   ElasticNetCV    EN w/Cross Validation
        rho             Dampening parameter(s)
        eps=1e-3        min(alpha)/max(alpha)
        n_alphas        Number of alphas
        alphas          List of alphas
        max_iter        Maximum iterations
        tol             Optimization tolerance
        cv=3            Cross validation folds

    "LA"    Lars            Least Angle Regression
        n_nonzero_coefs Number of non-zero coefficients
        eps             Cholesky regularization

    "LAC"   LarsCV          LAR w/Cross Validation
        max_iter        Maximum iterations
        cv=5            Cross validation folds
        max_n_alphas    Max points for residuals in cv

    "LAS"   Lasso           Least Absolute Shrinkage and
                            Selection Operator
        alpha=1.0       Dampening parameter
        max_iter        Maximum iterations
        tol             Optimization tolerance

    "LASC"  LassoCV         LAS w/Cross Validation
        eps=1e-3        min(alpha)/max(alpha)
        n_alphas        Number of alphas
        alphas          List of alphas
        max_iter        Maximum iterations
        tol             Optimization tolerance
        cv=3            Cross validation folds

    "LL"    LassoLars       Lasso and Lars model
        max_iter        Maximum iterations
        eps             Cholesky regularization

    "LLC"   LassoLarsCV     LL w/Cross Validation
        max_iter        Maximum iterations
        cv=5            Cross validation folds
        max_n_alphas    Max points for residuals in cv
        eps             Cholesky regularization

    "LLIC"  LassoLarsIC     LL w/AIC or BIC
        criterion       "AIC" or "BIC" criterion
        max_iter        Maximum iterations
        eps             Cholesky regularization

    "OMP"   OrthogonalMatchingPursuit
        n_nonzero_coefs Number of non-zero coefficients
        tol             Max residual norm (instead of non-zero coef)

    Local methods

    Key     Description
    ---     -----------
    "LS"    Ordenary Least Squares

    "T"     Ridge Regression/Tikhonov Regularization
        order           Order of regularization (or custom matrix)
        alpha           Dampning parameter (else estimated from gcv)

    "TC"    T w/Cross Validation
        order           Order of regularization (or custom matrix)
        alpha           Dampning parameter (else estimated from gcv)


Returns
-------
R[, uhat]

R : Poly
    Fitted polynomial with `R.shape=u.shape[1:]` and `R.dim=D`.
uhat : np.ndarray
    The Fourier coefficients in the estimation.

Examples
--------
>>> P = cp.Poly([1, x, y])
>>> s = [[-1,-1,1,1], [-1,1,-1,1]]
>>> u = [0,1,1,2]
>>> print fit_regression(P, s, u)
0.5q1+0.5q0+1.0

    """

    x = np.array(x)
    if len(x.shape)==1:
        x = x.reshape(1, *x.shape)
    u = np.array(u)

    Q = P(*x).T
    shape = u.shape[1:]
    u = u.reshape(u.shape[0], int(np.prod(u.shape[1:])))

    rule = rule.upper()

    # Local rules
    if rule=="LS":
        uhat = la.lstsq(Q, u)[0]

    elif rule=="T":
        uhat, alphas = rlstsq(Q, u, kws.get("order",0),
                kws.get("alpha", None), False, True)

    elif rule=="TC":
        uhat = rlstsq(Q, u, kws.get("order",0),
                kws.get("alpha", None), True)

    else:

        # Scikit-learn wrapper
        try:
            _ = lm
        except:
            raise NotImplementedError(
                    "sklearn not installed")

        if rule=="BARD":
            solver = lm.ARDRegression(fit_intercept=False,
                    copy_X=False, **kws)

        elif rule=="BR":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.BayesianRidge(**kws)

        elif rule=="EN":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.ElasticNet(**kws)

        elif rule=="ENC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.ElasticNetCV(**kws)

        elif rule=="LA": # success
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.Lars(**kws)

        elif rule=="LAC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.LarsCV(**kws)

        elif rule=="LAS":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.Lasso(**kws)

        elif rule=="LASC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.LassoCV(**kws)

        elif rule=="LL":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.LassoLars(**kws)

        elif rule=="LLC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.LassoLarsCV(**kws)

        elif rule=="LLIC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = lm.LassoLarsIC(**kws)

        elif rule=="OMP":
            solver = lm.OrthogonalMatchingPursuit(**kws)

        uhat = solver.fit(Q, u).coef_

    u = u.reshape(u.shape[0], *shape)

    R = po.sum((P*uhat.T), -1)
    R = po.reshape(R, shape)

    if retall==1:
        return R, uhat
    elif retall==2:
        if rule=="T":
            return R, uhat, Q, alphas
        return R, uhat, Q
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
        basis = po.basis(0, n, dim)
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

    for fold in xrange(folds):

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
>>> func = lambda q: q[0]*q[1]
>>> poly = cp.basis(0,2,2)
>>> dist = cp.J(cp.Uniform(0,1), cp.Uniform(0,1))
>>> res = cp.fit_adaptive(func, poly, dist, budget=100)
>>> print res
    """

    if bufname:
        func = lazy_eval(func, load=bufname)

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

    out = po.transpose(po.sum(poly*val2.T, -1))

    if retall:
        return out, val2, val1, err2, err1
    return val2




if __name__=="__main__":
    import numpy as np
    import __init__ as cp
    import doctest
    x, y = cp.variable(2)

    doctest.testmod()

