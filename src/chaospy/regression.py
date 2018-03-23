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
    [[ 0.67448975 -1.15034938  0.31863936 -0.31863936]
     [-1.42607687 -1.02007623 -0.73631592 -0.50240222]]

- A function evaluated using the nodes generated in the second step::

    >>> def model_solver(param):
    ...     return [param[0]*param[1], param[0]*np.e**-param[1]+1]
    >>> solves = [model_solver(sample) for sample in samples.T]
    >>> print(np.around(solves[:4], 8))
    [[-0.96187423  3.80745414]
     [ 1.17344406 -2.19038608]
     [-0.23461924  1.66539168]
     [ 0.16008512  0.47338898]]

- Bring it all together using `~chaospy.collocation.fit_regression`::

    >>> approx_model = cp.fit_regression(
    ...      orthogonal_expansion, samples, solves)
    >>> print(cp.around(approx_model, 5))
    [q0q1, 0.0478q0^2-1.4354q0q1+0.1108q1^2+1.22377q0-0.0907q1+0.93973]

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
used when fitting the samples to the polynomials:

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

In addition, the rule can be any initialized regression model from sklearn.
For example, if one wants to implement orthogonal matching pursuit
:cite:`mallat_matching_1993`, it can for example be implemented as follows::
. It forces the
result to have at most one non-zero coefficient. To implement it use the
keyword ``rule="OMP"``, and to force the number of coefficients to be
for example 1: ``n_nonzero_coefs=1``. In practice::

    >>> from sklearn.linear_model import OrthogonalMatchingPursuit
    >>> omp = OrthogonalMatchingPursuit(fit_intercept=False, n_nonzero_coefs=1)
    >>> approx_model = cp.fit_regression(
    ...     orthogonal_expansion, samples, solves, rule=omp)
    >>> print(cp.around(approx_model, 8))
    [3.46375077q0q1, 11.63750715]

Note that the option `fit_intercept=False`. This is a prerequisite for
``sklearn`` to be compatible with ``chaospy``.
"""
import numpy as np
from scipy import linalg, optimize

import chaospy as cp

__all__ = ("fit_regression", "rlstsq")


def fit_regression(
        polynomials,
        abscissas,
        evals,
        rule="LS",
        retall=False,
        order=0,
        alpha=None,
):
    """
    Fit a polynomial chaos expansion using linear regression.

    Args:
        polynomials (Poly):
            Polynomial expansion with `polynomials.shape=(M,)` and
            `polynomials.dim=D`.
        abscissas (array_like):
            Collocation nodes with `abscissas.shape=(D,K)`.
        evals (array_like):
            Model evaluations with `len(evals)=K`.
        retall (bool):
            If True return Fourier coefficients in addition to R.
        order (int):
            Tikhonov regularization order.
        alpha (float):
            Dampning parameter for the Tikhonov regularization. Calculated
            automatically if omitted.

    Returns:
        (Poly, np.ndarray):
            Fitted polynomial with `R.shape=evals.shape[1:]` and `R.dim=D`. The
            Fourier coefficients in the estimation.

    Examples:
        >>> x, y = cp.variable(2)
        >>> polynomials = cp.Poly([1, x, y])
        >>> abscissas = [[-1,-1,1,1], [-1,1,-1,1]]
        >>> evals = [0,1,1,2]
        >>> print(cp.around(cp.fit_regression(polynomials, abscissas, evals), 14))
        0.5q0+0.5q1+1.0
    """
    abscissas = np.asarray(abscissas)
    if len(abscissas.shape) == 1:
        abscissas = abscissas.reshape(1, *abscissas.shape)
    evals = np.array(evals)

    poly_evals = polynomials(*abscissas).T
    shape = evals.shape[1:]
    evals = evals.reshape(evals.shape[0], int(np.prod(evals.shape[1:])))

    if isinstance(rule, str):
        rule = rule.upper()

    if rule == "LS":
        uhat = linalg.lstsq(poly_evals, evals)[0]

    elif rule == "T":
        uhat = rlstsq(poly_evals, evals, order=order, alpha=alpha, cross=False)

    elif rule == "TC":
        uhat = rlstsq(poly_evals, evals, order=order, alpha=alpha, cross=True)

    else:

        from sklearn.linear_model.base import LinearModel
        assert isinstance(rule, LinearModel)
        uhat = rule.fit(poly_evals, evals).coef_.T

    evals = evals.reshape(evals.shape[0], *shape)

    approx_model = cp.poly.sum((polynomials*uhat.T), -1)
    approx_model = cp.poly.reshape(approx_model, shape)

    if retall == 1:
        return approx_model, uhat
    elif retall == 2:
        return approx_model, uhat, poly_evals
    return approx_model


def rlstsq(coef_mat, ordinate, order=1, alpha=None, cross=False):
    """
    Least Squares Minimization using Tikhonov regularization.

    Includes method for robust generalized cross-validation.

    Args:
        coef_mat (array_like):
            Coefficient matrix with shape (M,N).
        ordinate (array_like, shape (M,) or (M, K)):
            Ordinate or "dependent variable" values with shape (M,) or (M, K).
            If `ordinate` is two-dimensional, the least-squares solution is
            calculated for each of the `K` columns of `ordinate`.
        order (int, array_like):
            If int, it is the order of Tikhonov regularization.  If array_like,
            it will be used as regularization matrix.
        alpha (float, optional):
            Lower threshold for the dampening parameter. The real value is
            calculated using generalised cross validation.
        cross (bool, optional):
            Use cross validation to estimate alpha value.
    """
    coef_mat = np.array(coef_mat)
    ordinate = np.array(ordinate)
    dim1, dim2 = coef_mat.shape

    if cross:
        out = np.empty((dim1, dim2) + ordinate.shape[1:])
        coef_mat_ = np.empty((dim1-1, dim2))
        ordinate_ = np.empty((dim1-1,) + ordinate.shape[1:])
        for i in range(dim1):
            coef_mat_[:i] = coef_mat[:i]
            coef_mat_[i:] = coef_mat[i+1:]
            ordinate_[:i] = ordinate[:i]
            ordinate_[i:] = ordinate[i+1:]
            out[i] = rlstsq(coef_mat_, ordinate_, order, alpha, False)

        return np.median(out, 0)

    if order == 0:
        tikhmat = np.eye(dim2)

    elif order == 1:
        tikhmat = np.zeros((dim2-1, dim2))
        tikhmat[:, :-1] -= np.eye(dim2-1)
        tikhmat[:, 1:] += np.eye(dim2-1)

    elif order == 2:
        tikhmat = np.zeros((dim2-2, dim2))
        tikhmat[:, :-2] += np.eye(dim2-2)
        tikhmat[:, 1:-1] -= 2*np.eye(dim2-2)
        tikhmat[:, 2:] += np.eye(dim2-2)

    elif order is None:
        tikhmat = np.zeros(1)

    else:
        tikhmat = np.array(order)
        assert tikhmat.shape[-1] == dim2 or tikhmat.shape in ((), (1,))

    if alpha is None and order is not None:

        gamma = 0.1

        def rgcv_error(alpha):
            """Calculate Tikhonov dampening parameter."""
            if alpha <= 0:
                return np.inf
            coef_mat_ = np.dot(
                coef_mat.T, coef_mat)+alpha*(np.dot(tikhmat.T, tikhmat))
            try:
                coef_mat_ = np.dot(linalg.inv(coef_mat_), coef_mat.T)
            except linalg.LinAlgError:
                return np.inf

            abscissas = np.dot(coef_mat_, ordinate)
            res2 = np.sum((np.dot(coef_mat, abscissas)-ordinate)**2)
            coef_mat_2 = np.dot(coef_mat, coef_mat_)
            skew = dim1*res2/np.trace(np.eye(dim1)-coef_mat_2)**2
            mu2 = np.sum(coef_mat_2*coef_mat_2.T)/dim1

            return (gamma + (1-gamma)*mu2)*skew

        alpha = optimize.fmin(rgcv_error, 1, disp=0)

    out = linalg.inv(
        np.dot(coef_mat.T, coef_mat) + alpha*np.dot(tikhmat.T, tikhmat))
    out = np.dot(out, np.dot(coef_mat.T, ordinate))
    return out


if __name__ == "__main__":
    import doctest
    doctest.testmod()
