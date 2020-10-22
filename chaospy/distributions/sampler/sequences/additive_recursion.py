"""Additive recursion sequence."""
import numpy


def generalized_golden_ratio(dim):
    """
    Using nested radical formula to calculate generalized golden ratio.

    Args:
        dim (int):
            The number of dimension the ratio is to be used in.

    Returns:
        (float):
            The generalize golden ratio for dimension `dim`.

    Examples:
        >>> generalized_golden_ratio(1)
        1.618033988749895
        >>> generalized_golden_ratio(2)
        1.324717957244746
        >>> generalized_golden_ratio(100)
        1.0069208854344955
    """
    out = 1.7
    out_ = 1.
    while out != out_:
        out, out_ = out_, (1+out)**(1./(dim+1))
    return out_


def create_additive_recursion_samples(order, dim=1, seed=0.5, alpha=None):
    """
    Create samples from the additive recursion sequence.

    Args:
        order (int):
            The number of samples to produce.
        dim (int):
            The number of dimensions in the sequence.
        seed (float):
            Random seed to be used in the sequence.
        alpha (Optional[Sequence[float]]):
            The (irrational) numbers used to create additive recursion.
            If omitted, inverse of generalized golde-ratio values will be used.

    Returns:
        (numpy.ndarray):
            Samples from the additive recursion sequence,
            with shape `(dim, order)`.

    Examples:
        >>> chaospy.create_additive_recursion_samples(5, 1).round(4)
        array([[0.118 , 0.7361, 0.3541, 0.9721, 0.5902]])
        >>> chaospy.create_additive_recursion_samples(5, 2).round(4)
        array([[0.2549, 0.0098, 0.7646, 0.5195, 0.2744],
               [0.0698, 0.6397, 0.2095, 0.7794, 0.3492]])
        >>> chaospy.create_additive_recursion_samples(5, 3).round(4)
        array([[0.3192, 0.1383, 0.9575, 0.7767, 0.5959],
               [0.171 , 0.8421, 0.5131, 0.1842, 0.8552],
               [0.0497, 0.5994, 0.1491, 0.6988, 0.2485]])

    """
    assert isinstance(dim, int) and dim > 0
    assert 0 <= seed < 1
    if alpha is None:
        phi = generalized_golden_ratio(dim)
        alpha = (1./phi)**numpy.arange(1, dim+1) % 1
    assert isinstance(alpha, numpy.ndarray)
    assert alpha.shape == (dim,)
    return (seed+numpy.outer(alpha, numpy.arange(1, order+1))) % 1
