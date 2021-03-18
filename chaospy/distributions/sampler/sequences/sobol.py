"""
Generates samples from the Sobol sequence.

Papers::

    Antonov, Saleev,
    USSR Computational Mathematics and Mathematical Physics,
    Volume 19, 1980, pages 252 - 256.

    Paul Bratley, Bennett Fox,
    Algorithm 659:
    Implementing Sobol's Quasirandom Sequence Generator,
    ACM Transactions on Mathematical Software,
    Volume 14, Number 1, pages 88-100, 1988.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

    Ilya Sobol,
    USSR Computational Mathematics and Mathematical Physics,
    Volume 16, pages 236-242, 1977.

    Ilya Sobol, Levitan,
    The Production of Points Uniformly Distributed in a Multidimensional
    Cube (in Russian),
    Preprint IPM Akad. Nauk SSSR,
    Number 40, Moscow 1976.

"""
import math

import numpy

from .sobol_constants import DIM_MAX, LOG_MAX, POLY, SOURCE_SAMPLES


def create_sobol_samples(order, dim, seed=1):
    """
    Generates samples from the Sobol sequence.

    Sobol sequences (also called LP_T sequences or (t, s) sequences in base 2)
    are an example of quasi-random low-discrepancy sequences. They were first
    introduced by the Russian mathematician Ilya M. Sobol in 1967.

    These sequences use a base of two to form successively finer uniform
    partitions of the unit interval and then reorder the coordinates in each
    dimension.

    Args:
        order (int):
            Number of unique samples to generate.
        dim (int):
            Number of spacial dimensions. Must satisfy ``0 < dim < 1111``.
        seed (int):
            Starting seed. Non-positive values are treated as 1. If omitted,
            consecutive samples are used.

    Returns:
        (numpy.ndarray):
            Quasi-random vector with ``shape == (dim, order)``.

    Notes:
        Implementation based on the initial work of Sobol
        :cite:`sobol_distribution_1967`. This implementation is based on the
        work of `Burkardt
        <https://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html>`_.

    Examples:
        >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
        >>> samples = distribution.sample(3, rule="sobol")
        >>> samples.round(4)
        array([[0.5 , 0.75, 0.25],
               [0.5 , 0.25, 0.75]])
        >>> samples = distribution.sample(4, rule="sobol")
        >>> samples.round(4)
        array([[0.5  , 0.75 , 0.25 , 0.375],
               [0.5  , 0.25 , 0.75 , 0.375]])

    """
    assert 0 < dim < DIM_MAX, "dim in [1, 1111]"

    # Initialize row 1 of V.
    samples = SOURCE_SAMPLES.copy()
    samples[0, 0:LOG_MAX] = 1

    # Initialize the remaining rows of V.
    for idx in range(1, dim):

        # The bits of the integer POLY(I) gives the form of polynomial:
        degree = int(math.log(POLY[idx], 2))

        # Expand this bit pattern to separate components:
        includ = numpy.array([val == "1" for val in bin(POLY[idx])[-degree:]])

        #Calculate the remaining elements of row I as explained
        #in Bratley and Fox, section 2.
        for idy in range(degree+1, LOG_MAX+1):
            newv = samples[idx, idy-degree-1].item()
            base = 1
            for idz in range(1, degree+1):
                base *= 2
                if includ[idz-1]:
                    newv = newv ^ base * samples[idx, idy-idz-1].item()
            samples[idx, idy-1] = newv

    samples = samples[:dim]

    # Multiply columns of V by appropriate power of 2.
    samples *= 2**(numpy.arange(LOG_MAX, 0, -1, dtype=int))

    #RECIPD is 1/(common denominator of the elements in V).
    recipd = 0.5**(LOG_MAX+1)
    lastq = numpy.zeros(dim, dtype=int)

    seed = int(seed) if seed > 1 else 1

    for seed_ in range(seed):
        lowbit = len(bin(seed_)[2:].split("0")[-1])
        lastq[:] = lastq ^ samples[:, lowbit]

    #Calculate the new components of QUASI.
    quasi = numpy.empty((dim, order))
    for idx in range(order):
        lowbit = len(bin(seed+idx)[2:].split("0")[-1])
        quasi[:, idx] = lastq * recipd
        lastq[:] = lastq ^ samples[:, lowbit]

    return quasi
