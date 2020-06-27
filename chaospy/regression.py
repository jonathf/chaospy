"""
Point collocation method, or regression based polynomial chaos expansion builds
open the idea of fitting a polynomial chaos expansion to a set of generated
samples and evaluations. The experiment can be done as follows:

- Select a :ref:`distributions`::

    >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)

- Generate :ref:`orthogonality`::

    >>> orthogonal_expansion = chaospy.orth_ttr(2, distribution)
    >>> orthogonal_expansion
    polynomial([1.0, q1, q0, q1**2-1.0, q0*q1, q0**2-1.0])

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
    polynomial([q0*q1, 0.11*q1**2-1.44*q0*q1+0.05*q0**2-0.09*q1+1.22*q0+0.94])

In this example, the number of collocation points is selected to be twice the
number of unknown coefficients :math:`N+1`. Changing this is obviously
possible. When the number of parameter is equal the number of unknown, the, the
polynomial approximation becomes an interpolation method and overlap with
Lagrange polynomials. If the number of samples are fewer than the number of
unknown, classical least squares can not be used. Instead it possible to use
methods for doing estimation with too few samples.
"""
import logging
import numpy
from scipy import linalg

import numpoly
import chaospy


def fit_regression(
        polynomials,
        abscissas,
        evals,
        model=None,
        retall=False,
):
    """
    Fit a polynomial chaos expansion using linear regression.

    Args:
        polynomials (numpoly.ndpoly):
            Polynomial expansion with ``polynomials.shape == (M,)`` and
            `polynomials.dim=D`.
        abscissas (numpy.ndarray):
            Collocation nodes with ``abscissas.shape == (D, K)``.
        evals (numpy.ndarray):
            Model evaluations with ``len(evals) == K``.
        model (Optional[sklearn.base.BaseEstimator]):
            By default regression is done using the classical least-square
            method. However, if provided, and `sklearn` regression model can be
            used instead.
        retall (bool):
            If True return Fourier coefficients in addition to R.

    Returns:
        (chaospy.ndpoly, numpy.ndarray):
            Fitted polynomial with ``R.shape=evals.shape[1:]`` and ``R.dim=D``.
            The Fourier coefficients in the estimation.

    Examples:
        >>> x, y = chaospy.variable(2)
        >>> polynomials = chaospy.polynomial([1, x, y])
        >>> abscissas = [[-1,-1,1,1], [-1,1,-1,1]]
        >>> evals = [0,1,1,2]
        >>> chaospy.fit_regression(polynomials, abscissas, evals).round(14)
        polynomial(0.5*q1+0.5*q0+1.0)

    """
    logger = logging.getLogger(__name__)
    abscissas = numpy.asarray(abscissas)
    if len(abscissas.shape) == 1:
        abscissas = abscissas.reshape(1, *abscissas.shape)
    evals = numpy.array(evals)

    poly_evals = polynomials(*abscissas).T
    shape = evals.shape[1:]
    if shape:
        evals = evals.reshape(evals.shape[0], int(numpy.prod(evals.shape[1:])))

    if model is None:
        uhat = linalg.lstsq(poly_evals, evals)[0]

    else:
        try:
            from sklearn.base import BaseEstimator
        except ImportError:
            raise ValueError(
                "arg model != None requires that scikit-learn is installed")

        if not isinstance(model, BaseEstimator):
            raise ValueError("model not recognized; "
                             "Optional[sklearn.base.BaseEstimator] expected")
        if hasattr(model, "fit_intercept"):
            assert not model.fit_intercept, (
                "model %s must have fit_intercept=False" % model.__class__.__name__)
        uhat = model.fit(poly_evals, evals).coef_.T

    if shape:
        evals = evals.reshape(evals.shape[0], *shape)

    approx_model = numpoly.sum((polynomials*uhat.T), -1)
    approx_model = approx_model.reshape(shape)

    if retall == 1:
        return approx_model, uhat
    if retall == 2:
        return approx_model, uhat, poly_evals
    return approx_model
