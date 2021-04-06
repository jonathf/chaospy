r"""Create Gaussian quadrature nodes and weights."""
import chaospy

from .utils import combine_quadrature


def gaussian(
        order,
        dist,
        recurrence_algorithm="stieltjes",
        rule="clenshaw_curtis",
        tolerance=1e-10,
        scaling=3,
        n_max=5000,
):
    """
    Create Gaussian quadrature nodes and weights.

    Generating Gaussian quadrature by first generating so called *three terms
    recurrence* coefficients using one various different algorithms. The
    coefficients are them converted to abscissas and weights by constructing
    lower banded Jacobi matrix and calculating eigenvalues and eigenvectors,
    which can be directly translated to abscissas and weights. Construction of
    the coefficients is potentially numerically unstable, there for multiple
    algorithms exists.

    Args:
        order (int):
            The order of the quadrature.
        dist (chaospy.Distribution):
            The distribution which density will be used as weight function.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights.
        rule (str):
            In the case of ``lanczos`` or ``stieltjes``, defines the
            proxy-integration scheme.
        tolerance (float):
            The allowed relative error in norm between two quadrature orders
            before method assumes convergence.
        scaling (float):
            A multiplier the adaptive order increases with for each step
            quadrature order is not converged. Use 0 to indicate unit
            increments.
        n_max (int):
            The allowed number of quadrature points to use in approximation.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            Gaussian quadrature abscissas and weights with shapes
            ``(len(dist), order+1)`` and ``(order+1,)`` respectively.

    Raises:
        NotImplementedError:
            In the case of ``analytical`` three terms recurrence algorithm,
            error is raised if the distribution does not implement the feature.
            coefficients.
        numpy.linalg.LinAlgError:
            For non-canonical random variables, the construction might fail
            because of illegal numerical operations.

    Notes:
        Quadrature constructed as outlined by Walter Gautschi
        :cite:`gautschi_construction_1968`.

    Examples:
        >>> distribution = chaospy.Normal(0, 1)
        >>> abscissas, weights = chaospy.quadrature.gaussian(5, distribution)
        >>> abscissas.round(4)
        array([[-3.3243, -1.8892, -0.6167,  0.6167,  1.8892,  3.3243]])
        >>> weights.round(4)
        array([0.0026, 0.0886, 0.4088, 0.4088, 0.0886, 0.0026])
        >>> distribution = chaospy.J(chaospy.Uniform(), chaospy.Normal())
        >>> abscissas, weights = chaospy.quadrature.gaussian(2, distribution)
        >>> abscissas.round(2)
        array([[ 0.11,  0.11,  0.11,  0.5 ,  0.5 ,  0.5 ,  0.89,  0.89,  0.89],
               [-1.73,  0.  ,  1.73, -1.73,  0.  ,  1.73, -1.73,  0.  ,  1.73]])
        >>> weights.round(3)
        array([0.046, 0.185, 0.046, 0.074, 0.296, 0.074, 0.046, 0.185, 0.046])

    """
    coefficients = chaospy.construct_recurrence_coefficients(
        order=order,
        dist=dist,
        recurrence_algorithm=recurrence_algorithm,
        rule=rule,
        tolerance=tolerance,
        scaling=scaling,
        n_max=n_max,
    )
    abscissas, weights = chaospy.coefficients_to_quadrature(coefficients)
    return combine_quadrature(abscissas, weights)
