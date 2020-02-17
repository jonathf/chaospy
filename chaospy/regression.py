"""
Point collocation method, or regression based polynomial chaos expansion builds
open the idea of fitting a polynomial chaos expansion to a set of generated
samples and evaluations. The experiment can be done as follows:

- Select a :ref:`distributions`::

    >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)

- Generate :ref:`orthogonality`::

    >>> orthogonal_expansion = chaospy.orth_ttr(2, distribution)
    >>> orthogonal_expansion
    polynomial([1.0, q1, q0, -1.0+q1**2, q0*q1, -1.0+q0**2])

- Generate samples using :ref:`sampling` (or alternative abscissas from
  :ref:`quadrature`)::

    >>> samples = distribution.sample(
    ...     2*len(orthogonal_expansion), rule="hammersley")
    >>> samples[:, :4]
    array([[ 0.67448975, -1.15034938,  0.31863936, -0.31863936],
           [-1.42607687, -1.02007623, -0.73631592, -0.50240222]])

- A function evaluated using the nodes generated in the second step::

    >>> def model_solver(param):
    ...     return [param[0]*param[1], param[0]*numpy.e**-param[1]+1]
    >>> solves = numpy.array([model_solver(sample) for sample in samples.T])
    >>> solves[:4].round(8)
    array([[-0.96187423,  3.80745414],
           [ 1.17344406, -2.19038608],
           [-0.23461924,  1.66539168],
           [ 0.16008512,  0.47338898]])

- Bring it all together using `~chaospy.regression.fit_regression`::

    >>> approx_model = chaospy.fit_regression(
    ...      orthogonal_expansion, samples, solves)
    >>> approx_model.round(2)
    polynomial([q0*q1, 0.94-0.09*q1+0.11*q1**2+1.22*q0-1.44*q0*q1+0.05*q0**2])

In this example, the number of collocation points is selected to be twice the
number of unknown coefficients :math:`N+1`. Changing this is obviously
possible. When the number of parameter is equal the number of unknown, the, the
polynomial approximation becomes an interpolation method and overlap with
Lagrange polynomials. If the number of samples are fewer than the number of
unknown, classical least squares can not be used. Instead it possible to use
methods for doing estimation with too few samples.

The function :func:`~chaospy.regression.fit_regression` also takes an optional
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

In addition, the rule can be any initialized regression model from
``scipit-learn``. For example, if one wants to implement orthogonal matching
pursuit : It forces the result to have at most one non-zero coefficient. To
implement it use the keyword ``rule="OMP"``, and to force the number of
coefficients to be for example 1: ``n_nonzero_coefs=1``. In practice::

    >>> from sklearn.linear_model import OrthogonalMatchingPursuit  # doctest: +SKIP
    >>> omp = OrthogonalMatchingPursuit(fit_intercept=False, n_nonzero_coefs=1)  # doctest: +SKIP
    >>> approx_model = chaospy.fit_regression(  # doctest: +SKIP
    ...     orthogonal_expansion, samples, solves, rule=omp)
    >>> approx_model.round(2)
    polynomial([q0*q1, 0.94-0.09*q1+0.11*q1**2+1.22*q0-1.44*q0*q1+0.05*q0**2])

Note that the option ``fit_intercept=False``. This is a prerequisite for
``scikit-learn`` to be compatible with ``chaospy``.
"""
import numpy
from scipy import linalg

import chaospy


def fit_regression(
        polynomials,
        abscissas,
        evals,
        rule="LS",
        retall=False,
        order=0,
        alpha=-1,
):
    """
    Fit a polynomial chaos expansion using linear regression.

    Args:
        polynomials (chaospy.poly.ndpoly):
            Polynomial expansion with ``polynomials.shape=(M,)`` and
            `polynomials.dim=D`.
        abscissas (numpy.ndarray):
            Collocation nodes with ``abscissas.shape == (D, K)``.
        evals (numpy.ndarray):
            Model evaluations with ``len(evals)=K``.
        retall (bool):
            If True return Fourier coefficients in addition to R.
        order (int):
            Tikhonov regularization order.
        alpha (float):
            Dampning parameter for the Tikhonov regularization. Calculated
            automatically if negative.

    Returns:
        (chaospy.poly.ndpoly, numpy.ndarray):
            Fitted polynomial with ``R.shape=evals.shape[1:]`` and ``R.dim=D``.
            The Fourier coefficients in the estimation.

    Examples:
        >>> x, y = chaospy.variable(2)
        >>> polynomials = chaospy.polynomial([1, x, y])
        >>> abscissas = [[-1,-1,1,1], [-1,1,-1,1]]
        >>> evals = [0,1,1,2]
        >>> chaospy.fit_regression(polynomials, abscissas, evals).round(14)
        polynomial(1.0+0.5*q1+0.5*q0)
    """
    abscissas = numpy.asarray(abscissas)
    if len(abscissas.shape) == 1:
        abscissas = abscissas.reshape(1, *abscissas.shape)
    evals = numpy.array(evals)

    poly_evals = polynomials(*abscissas).T
    shape = evals.shape[1:]
    evals = evals.reshape(evals.shape[0], int(numpy.prod(evals.shape[1:])))

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

    approx_model = chaospy.poly.sum((polynomials*uhat.T), -1)
    approx_model = approx_model.reshape(shape)

    if retall == 1:
        return approx_model, uhat
    elif retall == 2:
        return approx_model, uhat, poly_evals
    return approx_model


def rlstsq(coef_mat, ordinate, order=1, alpha=-1, cross=False):
    """
    Least Squares Minimization using Tikhonov regularization.

    Includes method for robust generalized cross-validation.

    Args:
        coef_mat (numpy.ndarray):
            Coefficient matrix with shape ``(M, N)``.
        ordinate (numpy.ndarray):
            Ordinate or "dependent variable" values with shape ``(M,)`` or
            ``(M, K)``. If ``ordinate`` is two-dimensional, the least-squares
            solution is calculated for each of the ``K`` columns of
            ``ordinate``.
        order (int, numpy.ndarray):
            If int, it is the order of Tikhonov regularization. If
            `numpy.ndarray`, it will be used as regularization matrix.
        alpha (float):
            Lower threshold for the dampening parameter. The real value is
            calculated using generalised cross validation.
        cross (bool):
            Use cross validation to estimate alpha value.
    """
    coef_mat = numpy.array(coef_mat)
    ordinate = numpy.array(ordinate)
    dim1, dim2 = coef_mat.shape

    if cross:
        out = numpy.empty((dim1, dim2) + ordinate.shape[1:])
        coef_mat_ = numpy.empty((dim1-1, dim2))
        ordinate_ = numpy.empty((dim1-1,) + ordinate.shape[1:])
        for i in range(dim1):
            coef_mat_[:i] = coef_mat[:i]
            coef_mat_[i:] = coef_mat[i+1:]
            ordinate_[:i] = ordinate[:i]
            ordinate_[i:] = ordinate[i+1:]
            out[i] = rlstsq(coef_mat_, ordinate_, order, alpha, False)

        return numpy.median(out, 0)

    if order == 0:
        tikhmat = numpy.eye(dim2)

    elif order == 1:
        tikhmat = numpy.zeros((dim2-1, dim2))
        tikhmat[:, :-1] -= numpy.eye(dim2-1)
        tikhmat[:, 1:] += numpy.eye(dim2-1)

    elif order == 2:
        tikhmat = numpy.zeros((dim2-2, dim2))
        tikhmat[:, :-2] += numpy.eye(dim2-2)
        tikhmat[:, 1:-1] -= 2*numpy.eye(dim2-2)
        tikhmat[:, 2:] += numpy.eye(dim2-2)

    elif order is None:
        tikhmat = numpy.zeros(1)

    else:
        tikhmat = numpy.array(order)
        assert tikhmat.shape[-1] == dim2 or tikhmat.shape in ((), (1,))

    if alpha < 0 and order is not None:

        gamma = 0.1

        def rgcv_error(alpha):
            """Calculate Tikhonov dampening parameter."""
            if alpha <= 0:
                return numpy.inf
            coef_mat_ = numpy.dot(
                coef_mat.T, coef_mat)+alpha*(numpy.dot(tikhmat.T, tikhmat))
            try:
                coef_mat_ = numpy.dot(linalg.inv(coef_mat_), coef_mat.T)
            except linalg.LinAlgError:
                return numpy.inf

            abscissas = numpy.dot(coef_mat_, ordinate)
            res2 = numpy.sum((numpy.dot(coef_mat, abscissas)-ordinate)**2)
            coef_mat_2 = numpy.dot(coef_mat, coef_mat_)
            skew = dim1*res2/numpy.trace(numpy.eye(dim1)-coef_mat_2)**2
            mu2 = numpy.sum(coef_mat_2*coef_mat_2.T)/dim1

            return (gamma + (1-gamma)*mu2)*skew

        alphas = 10.**-numpy.arange(0, 16)
        evals = numpy.array([rgcv_error(alpha) for alpha in alphas])
        alpha = alphas[numpy.argmin(evals)]

    out = linalg.inv(
        numpy.dot(coef_mat.T, coef_mat) + alpha*numpy.dot(tikhmat.T, tikhmat))
    out = numpy.dot(out, numpy.dot(coef_mat.T, ordinate))
    return out
