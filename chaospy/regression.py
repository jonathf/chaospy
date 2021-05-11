"""Fit a polynomial chaos expansion using linear regression."""
import numpy
import numpoly


def fit_regression(
        polynomials,
        abscissas,
        evals,
        model=None,
        retall=0,
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
        retall (int):
            What the function should return.
            0: only return fitted polynomials, with shape `evals.shape[1:]`.
            1: polynomials, and Fourier coefficients,
            2: polynomials, coefficients and polynomial evaluations.

    Returns:
        (chaospy.ndpoly, numpy.ndarray, numpy.ndarray):
            Returned value as determined by `retval`.

    Examples:
        >>> x, y = chaospy.variable(2)
        >>> polynomials = chaospy.polynomial([1, x, y])
        >>> abscissas = [[-1, -1, 1, 1],  [-1, 1, -1, 1]]
        >>> evals = [0, 1, 1, 2]
        >>> chaospy.fit_regression(polynomials, abscissas, evals).round(14)
        polynomial(0.5*q1+0.5*q0+1.0)
        >>> model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        >>> chaospy.fit_regression(
        ...     polynomials, abscissas, evals, model=model).round(14)
        polynomial(0.5*q1+0.5*q0+1.0)

    """
    abscissas = numpy.atleast_2d(abscissas)
    assert abscissas.ndim == 2, "too many dimensions"

    polynomials = numpoly.aspolynomial(polynomials)

    evals = numpy.asarray(evals)
    assert abscissas.shape[-1] == len(evals)

    poly_evals = polynomials(*abscissas).T
    shape = evals.shape[1:]
    if shape:
        evals = evals.reshape(len(evals), -1)

    if model is None:
        uhat, _, _, _ = numpy.linalg.lstsq(poly_evals, evals, rcond=None)

    else:
        try:
            from sklearn.base import BaseEstimator
        except ImportError:  # pragma: no cover
            raise ValueError(
                "arg model != None requires that scikit-learn is installed")
        assert isinstance(model, BaseEstimator), (
            "model not recognized; "
            "Optional[sklearn.base.BaseEstimator] expected"
        )
        if hasattr(model, "fit_intercept"):
            assert not model.fit_intercept, (
                "requires %s(fit_intercept=False)" % model.__class__.__name__)
        uhat = numpy.transpose(model.fit(poly_evals, evals).coef_)

    approx_model = numpoly.sum((polynomials*uhat.T), -1).reshape(shape)
    choices = {0: approx_model,
               1: (approx_model, uhat),
               2: (approx_model, uhat, poly_evals)}
    return choices[retall]
