"""Example code used in the tutorials."""
import numpy
import chaospy

coordinates = numpy.linspace(0, 10, 1000)

distribution_init = chaospy.Normal(1.5, 0.2)
distribution_rate = chaospy.Uniform(0.1, 0.2)
distribution = chaospy.J(distribution_init, distribution_rate)

_t = coordinates[1:]
true_mean = numpy.hstack([1.5, 15*(numpy.e**(-0.1*_t)-numpy.e**(-0.2*_t))/_t])
true_variance = numpy.hstack([2.29, 11.45*(numpy.e**(-0.2*_t)-numpy.e**(-0.4*_t))/_t])-true_mean**2


def exponential_model(parameters, coordinates=coordinates):
    """
    Over simplistic exponential model function.

    Args:
        parameters (numpy.ndarray):
            Hyper-parameters defining the model initial conditions and
            exponential growth rate.
            Assumed to have ``parameters.shape == (2,)``.
        coordinates (numpy.ndarray):
            The spatio-temporal coordinates.

    Returns:
        (numpy.ndarray):
            Evaluation of the exponential model. Same shape as ``coordinates``.

    Examples:
        >>> evals = exponential_model(
        ...     parameters=(2, 4), coordinates=numpy.linspace(0, 1, 7))
        >>> evals.round(4)
        array([2.    , 1.0268, 0.5272, 0.2707, 0.139 , 0.0713, 0.0366])

    """
    param_init, param_rate = parameters
    return param_init*numpy.e**(-param_rate*coordinates)


def error_mean(prediction_mean, true_mean=true_mean):
    """
    How close the estimated mean is the to the true mean.

    Args:
        prediction_mean (numpy.ndarray):
            The estimated mean.
        true_mean (numpy.ndarray):
            The reference mean value. Must be same shape as
            ``prediction_mean``.

    Returns:
        (float):
            The mean absolute distance between predicted and true values.

    Examples:
        >>> predicted_mean = chaospy.Normal(0, 1).sample(100)
        >>> errors = error_mean(predicted_mean, true_mean=0)
        >>> errors.round(4)
        0.7864

    """
    return numpy.mean(numpy.abs(prediction_mean-true_mean))


def error_variance(predicted_variance, true_variance=true_variance):
    """
    How close the estimated variance is the to the true variance.

    Args:
        prediction_variance (numpy.ndarray):
            The estimated variance.
        true_variance (numpy.ndarray):
            The reference variance value. Must be same shape as
            ``prediction_variance``.

    Returns:
        (float):
            The mean absolute distance between predicted and true values.

    Examples:
        >>> predicted_var = chaospy.Exponential(1).sample(100)
        >>> errors = error_variance(predicted_var, true_variance=1)
        >>> errors.round(4)
        0.7068

    """
    return numpy.mean(numpy.abs(predicted_variance-true_variance))
