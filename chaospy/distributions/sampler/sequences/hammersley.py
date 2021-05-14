"""Create samples from the Hammersley set."""
import numpy

from .halton import create_halton_samples


def create_hammersley_samples(order, dim=1, burnin=-1, primes=()):
    """
    Create samples from the Hammersley set.

    The Hammersley set is equivalent to the Halton sequence, except for one
    dimension is replaced with a regular grid.

    Args:
        order (int):
            The order of the Hammersley sequence. Defines the number of samples.
        dim (int):
            The number of dimensions in the Hammersley sequence.
        burnin (int):
            Skip the first ``burnin`` samples. If negative, the maximum of
            ``primes`` is used.
        primes (tuple):
            The (non-)prime base to calculate values along each axis. If
            empty, growing prime values starting from 2 will be used.

    Returns:
        (numpy.ndarray):
            Hammersley set with ``shape == (dim, order)``.

    Examples:
        >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
        >>> samples = distribution.sample(3, rule="hammersley")
        >>> samples.round(4)
        array([[0.75 , 0.125, 0.625],
               [0.25 , 0.5  , 0.75 ]])
        >>> samples = distribution.sample(4, rule="hammersley")
        >>> samples.round(4)
        array([[0.75 , 0.125, 0.625, 0.375],
               [0.2  , 0.4  , 0.6  , 0.8  ]])

    """
    out = numpy.empty((dim, order), dtype=float)
    out[:max(dim-1, 1)] = create_halton_samples(
        order=order, dim=max(dim-1, 1), burnin=burnin, primes=primes)
    if dim > 1:
        out[dim-1] = numpy.linspace(0, 1, order+2)[1:-1]
    return out
