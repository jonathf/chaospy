"""
Point collcation method, or regression based polynomial chaos expansion builds
open the idea of fitting a polynomial chaos expansion to a set of generated
samples and evaluations. The experiement can be done as follows:

- Select a :ref:`distributions`::

      >>> distribution = cp.Iid(cp.Normal(0, 1), 2)

- Generate :ref:`orthogonality`::

      >>> orthogonal_expansion = cp.orth_ttr(2, distribution)
      >>> print(orthogonal_expansion)
      [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]

- Generate samples using :ref:`montecarlo` (or alternative absissas from
  :ref:`quadrature`)::

      >>> samples = distribution.sample(
      ...     2*len(orthogonal_expansion), rule="M")
      >>> print(samples[:, :4])
      [[-1.42607687 -1.02007623 -0.73631592 -0.50240222]
       [ 0.         -0.67448975  0.67448975 -1.15034938]]

- A function evaluated using the nodes generated in the second step::

      >>> def model_solver(param):
      ...     return [param[0]*param[1], param[0]*np.e**-param[1]+1]
      >>> solves = [model_solver(sample) for sample in samples.T]
      >>> print(np.around(solves[:4], 8))
      [[-0.         -0.42607687]
       [ 0.68803096 -1.00244135]
       [-0.49663754  0.62490868]
       [ 0.57793809 -0.58723759]]

- Bring it all together using `~chaospy.collocation.fit_regression`::

      >>> approx_model = cp.fit_regression(
      ...      orthogonal_expansion, samples, solves)
      >>> print(cp.around(approx_model, 5))
      [q0q1, 0.15275q0^2-1.23005q0q1+0.16104q1^2+1.214q0+0.044q1+0.86842]

In this example, the number of collocation points is selected to be twice the
number of unknown coefficients :math:`N+1`. This
follows the default outlined in :cite:`hosder_efficient_2007`. Changing this is
obviously possible. When the number of parameter is equal the number of
unknown, the, the polynomial approximation becomes an interpolation method and
overlap with Lagrange polynomials. If the number of samples are fewer than the
number of unknown, classical least squares can not be used.  Instead it
possible to use methods for doing estimation with too few samples.

The function :func:`~chaospy.collocation.fit_regression` also takes an optional
``rule`` keyword argument. It allows for the selection of regression method
used when fitting the samples to the polynomials.  One example there is
orthogonal matching pursuit :cite:`mallat_matching_1993`. It forces the
result to have at most one non-zero coefficient. To implement it use the
keyword ``rule="OMP"``, and to force the number of coefficients to be
for example 1: ``n_nonzero_coefs=1``. In practice::

   >>> approx_model = cp.fit_regression(
   ...     orthogonal_expansion, samples, solves,
   ...     rule="OMP", n_nonzero_coefs=1)
   >>> print(cp.around(approx_model, 8))
   [q0q1, 1.52536468q0]

Except for least squares, Tikhonov regularization with and without cross
validation, all the method listed is taken from ``sklearn`` software. All
optional arguments for various methods is covered in both
``sklearn.linear_model`` and in ``cp.fit_regression``.

The follwong methods uses scikits-learn as backend.
See `sklearn.linear_model` for more details.

+--------+---------------------------+-------------------------------------+
| Key    | Scikit-learn name         | Description                         |
|        | Parameters                |                                     |
+========+===========================+=====================================+
| "BARD" | ARDRegression             | Bayesian ARD Regression             |
|        | n_iter=300                | Maximum iterations                  |
|        | tol=1e-3                  | Optimization tolerance              |
|        | alpha_1=1e-6              | Gamma scale parameter               |
|        | alpha_2=1e-6              | Gamma inverse scale parameter       |
|        | lambda_1=1e-6             | Gamma shape parameter               |
|        | lambda_2=1e-6             | Gamma inverse scale parameter       |
|        | threshold_lambda=1e-4     | Upper pruning threshold             |
+--------+---------------------------+-------------------------------------+
| "BR"   | BayesianRidge             | Bayesian Ridge Regression           |
|        | n_iter=300                | Maximum iterations                  |
|        | tol=1e-3                  | Optimization tolerance              |
|        | alpha_1=1e-6              | Gamma scale parameter               |
|        | alpha_2=1e-6              | Gamma inverse scale parameter       |
|        | lambda_1=1e-6             | Gamma shape parameter               |
|        | lambda_2=1e-6             | Gamma inverse scale parameter       |
+--------+---------------------------+-------------------------------------+
| "EN"   | ElastiNet                 | Elastic Net                         |
|        | alpha=1.0                 | Dampening parameter                 |
|        | rho                       | Mixing parameter in [0,1]           |
|        | max_iter=300              | Maximum iterations                  |
|        | tol                       | Optimization tolerance              |
+--------+---------------------------+-------------------------------------+
| "ENC"  | ElasticNetCV              | EN w/Cross Validation               |
|        | rho                       | Dampening parameter(s)              |
|        | eps=1e-3                  | min(alpha)/max(alpha)               |
|        | n_alphas                  | Number of alphas                    |
|        | alphas                    | List of alphas                      |
|        | max_iter                  | Maximum iterations                  |
|        | tol                       | Optimization tolerance              |
|        | cv=3                      | Cross validation folds              |
+--------+---------------------------+-------------------------------------+
| "LA"   | Lars                      | Least Angle Regression              |
|        | n_nonzero_coefs           | Number of non-zero coefficients     |
|        | eps                       | Cholesky regularization             |
+--------+---------------------------+-------------------------------------+
| "LAC"  | LarsCV                    | LAR w/Cross Validation              |
|        | max_iter                  | Maximum iterations                  |
|        | cv=5                      | Cross validation folds              |
|        | max_n_alphas              | Max points for residuals in cv      |
+--------+---------------------------+-------------------------------------+
| "LAS"  | Lasso                     | Least Abs Shrink \& Select Operator |
|        | alpha=1.0                 | Dampening parameter                 |
|        | max_iter                  | Maximum iterations                  |
|        | tol                       | Optimization tolerance              |
+--------+---------------------------+-------------------------------------+
| "LASC" | LassoCV                   | LAS w/Cross Validation              |
|        | eps=1e-3                  | min(alpha)/max(alpha)               |
|        | n_alphas                  | Number of alphas                    |
|        | alphas                    | List of alphas                      |
|        | max_iter                  | Maximum iterations                  |
|        | tol                       | Optimization tolerance              |
|        | cv=3                      | Cross validation folds              |
+--------+---------------------------+-------------------------------------+
| "LL"   | LassoLars                 | Lasso and Lars model                |
|        | max_iter                  | Maximum iterations                  |
|        | eps                       | Cholesky regularization             |
+--------+---------------------------+-------------------------------------+
| "LLC"  | LassoLarsCV               | LL w/Cross Validation               |
|        | max_iter                  | Maximum iterations                  |
|        | cv=5                      | Cross validation folds              |
|        | max_n_alphas              | Max points for residuals in cv      |
|        | eps                       | Cholesky regularization             |
+--------+---------------------------+-------------------------------------+
| "LLIC" | LassoLarsIC               | LL w/AIC or BIC                     |
|        | criterion                 | "AIC" or "BIC" criterion            |
|        | max_iter                  | Maximum iterations                  |
|        | eps                       | Cholesky regularization             |
+--------+---------------------------+-------------------------------------+
| "OMP"  | OrthogonalMatchingPursuit | Orthogonal matching pursuit         |
|        | n_nonzero_coefs           | Number of non-zero coefficients     |
|        | tol                       | Max residual norm                   |
+--------+---------------------------+-------------------------------------+

There is also the following local methods:

+------------+----------------------------------------------+
| Key        | Description                                  |
| Parameters |                                              |
+============+==============================================+
| "LS"       | Ordenary Least Squares                       |
+------------+----------------------------------------------+
| "T"        | Ridge Regression/Tikhonov Regularization     |
| order      | Order of regularization (or custom matrix)   |
| alpha      | Dampning parameter (else estimated from gcv) |
+------------+----------------------------------------------+
| "TC"       | Tikhonov with Cross Validation               |
| order      | Order of regularization (or custom matrix)   |
| alpha      | Dampning parameter (else estimated from gcv) |
+------------+----------------------------------------------+
"""

__all__ = [
"fit_regression", "lstsq_cv", "rlstsq"
]

import numpy as np
from scipy import linalg, optimize

try:
    from sklearn import linear_model
except:
    pass

import chaospy as cp

def fit_regression(P, x, u, rule="LS", retall=False, **kws):
    """
    Fit a polynomial chaos expansion using linear regression.

    Args:
        P (Poly) : Polynomial expansion with `P.shape=(M,)` and `P.dim=D`.
        x (array_like) : Collocation nodes with `x.shape=(D,K)`.
        u (array_like) : Model evaluations with `len(u)=K`.
        retall (bool) : If True return Fourier coefficients in addition to R.
        rule (str) : Regression method used.

    Returns:
        (Poly, np.ndarray) : Fitted polynomial with `R.shape=u.shape[1:]` and
                `R.dim=D`. The Fourier coefficients in the estimation.

    Examples:
        >>> x, y = cp.variable(2)
        >>> P = cp.Poly([1, x, y])
        >>> s = [[-1,-1,1,1], [-1,1,-1,1]]
        >>> u = [0,1,1,2]
        >>> print(cp.around(cp.fit_regression(P, s, u), 14))
        0.5q0+0.5q1+1.0
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
        uhat = linalg.lstsq(Q, u)[0].T

    elif rule=="T":
        uhat, alphas = rlstsq(Q, u, kws.get("order",0),
                kws.get("alpha", None), False, True)
        uhat = uhat.T

    elif rule=="TC":
        uhat = rlstsq(Q, u, kws.get("order",0),
                kws.get("alpha", None), True)
        uhat = uhat.T

    else:

        # Scikit-learn wrapper
        try:
            _ = linear_model
        except:
            raise NotImplementedError(
                    "sklearn not installed")

        if rule=="BARD":
            solver = linear_model.ARDRegression(
                fit_intercept=False, copy_X=False, **kws)

        elif rule=="BR":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.BayesianRidge(**kws)

        elif rule=="EN":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.ElasticNet(**kws)

        elif rule=="ENC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.ElasticNetCV(**kws)

        elif rule=="LA": # success
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.Lars(**kws)

        elif rule=="LAC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.LarsCV(**kws)

        elif rule=="LAS":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.Lasso(**kws)

        elif rule=="LASC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.LassoCV(**kws)

        elif rule=="LL":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.LassoLars(**kws)

        elif rule=="LLC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.LassoLarsCV(**kws)

        elif rule=="LLIC":
            kws["fit_intercept"] = kws.get("fit_intercept", False)
            solver = linear_model.LassoLarsIC(**kws)

        elif rule=="OMP":
            solver = linear_model.OrthogonalMatchingPursuit(**kws)

        uhat = solver.fit(Q, u).coef_

    u = u.reshape(u.shape[0], *shape)

    R = cp.poly.sum((P*uhat), -1)
    R = cp.poly.reshape(R, shape)

    if retall==1:
        return R, uhat

    elif retall==2:
        if rule=="T":
            return R, uhat, Q, alphas
        return R, uhat, Q

    return R


def rlstsq(A, b, order=1, alpha=None, cross=False, retall=False):
    """
    Least Squares Minimization using Tikhonov regularization.

    Includes method for robust generalized cross-validation.

    Args:
        A (array_like, shape (M,N)) : "Coefficient" matrix.
        b (array_like, shape (M,) or (M, K)) : Ordinate or "dependent
                variable" values. If `b` is two-dimensional, the least-squares
                solution is calculated for each of the `K` columns of `b`.
        order (int, array_like) : If int, it is the order of Tikhonov
                regularization.  If array_like, it will be used as
                regularization matrix.
        alpha (float, optional) : Lower threshold for the dampening parameter.
                The real value is calculated using generalised cross
                validation.
        cross (bool) : Use cross validation
        retall (bool) : If True, return also estimated alpha-value
    """
    A = np.array(A)
    b = np.array(b)
    m,l = A.shape

    if cross:
        out = np.empty((m,l) + b.shape[1:])
        A_ = np.empty((m-1,l))
        b_ = np.empty((m-1,) + b.shape[1:])
        for i in range(m):
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
                A_ = np.dot(linalg.inv(A_), A.T)
            except linalg.LinAlgError:
                return np.inf
            x = np.dot(A_, b)
            res2 = np.sum((np.dot(A,x)-b)**2)
            K = np.dot(A, A_)
            V = m*res2/np.trace(np.eye(m)-K)**2
            mu2 = np.sum(K*K.T)/m

            return (gamma + (1-gamma)*mu2)*V

        alpha = optimize.fmin(rgcv_error, 1, disp=0)

    out = linalg.inv(np.dot(A.T,A) + alpha*np.dot(L.T, L))
    out = np.dot(out, np.dot(A.T, b))
    if retall:
        return out, alpha
    return out


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

    return linalg.lstsq(A, b)
