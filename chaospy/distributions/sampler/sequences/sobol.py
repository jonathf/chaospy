"""
Generates samples from the `Sobol sequence`_.

Sobol sequences (also called LP_T sequences or (t, s) sequences in base 2) are
an example of quasi-random low-discrepancy sequences. They were first
introduced by the Russian mathematician Ilya M. Sobol in 1967.

These sequences use a base of two to form successively finer uniform partitions
of the unit interval and then reorder the coordinates in each dimension.

Example usage
-------------

Standard usage::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    >>> samples = distribution.sample(3, rule="sobol")
    >>> print(numpy.around(samples, 4))
    [[0.5  0.75 0.25]
     [0.5  0.25 0.75]]
    >>> samples = distribution.sample(4, rule="sobol")
    >>> print(numpy.around(samples, 4))
    [[0.5   0.75  0.25  0.375]
     [0.5   0.25  0.75  0.375]]

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

.. _Sobel sequence: https://en.wikipedia.org/wiki/Sobol_sequence
"""
import math

import numpy

DIM_MAX = 40
LOG_MAX = 30

SOURCE_SAMPLES = numpy.zeros((DIM_MAX, LOG_MAX), dtype=int)
SOURCE_SAMPLES[0:40, 0] = 1
SOURCE_SAMPLES[2:40, 1] = (
    1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3,
    1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3
)
SOURCE_SAMPLES[3:40, 2] = (
    7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1, 5, 3, 3, 1, 7, 5, 1, 3,
    3, 7, 5, 1, 1, 5, 7, 7, 5, 1, 3, 3
)
SOURCE_SAMPLES[5:40, 3] = (
    1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15, 5, 3, 15, 7, 9, 13, 9,
    1, 11, 7, 5, 15, 1, 15, 11, 5, 3, 1, 7, 9
)
SOURCE_SAMPLES[7:40, 4] = (
    9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17, 1, 25, 29, 3, 31, 11, 5,
    23, 27, 19, 21, 5, 1, 17, 13, 7, 15, 9, 31, 9
)
SOURCE_SAMPLES[13:40, 5] = (
    37, 33, 7, 5, 11, 39, 63, 27, 17, 15, 23, 29, 3, 21, 13, 31, 25, 9, 49, 33,
    19, 29, 11, 19, 27, 15, 25
)
SOURCE_SAMPLES[19:40, 6] = (
    13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7, 59, 65, 21, 3, 113, 61,
    89, 45, 107
)
SOURCE_SAMPLES[37:40, 7] = (7, 23, 39)
POLY = (
    1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109, 103,
    115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171, 213, 191,
    253, 203, 211, 239, 247, 285, 369, 299
)


def create_sobol_samples(order, dim, seed=1):
    """
    Args:
        order (int):
            Number of unique samples to generate.
        dim (int):
            Number of spacial dimensions. Must satisfy ``0 < dim < 41``.
        seed (int):
            Starting seed. Non-positive values are treated as 1. If omitted,
            consecutive samples are used.

    Returns:
        (numpy.ndarray):
            Quasi-random vector with ``shape == (dim, order)``.
    """
    assert 0 < dim < DIM_MAX, "dim in [1, 40]"

    # Initialize row 1 of V.
    samples = SOURCE_SAMPLES.copy()
    maxcol = int(math.log(2**LOG_MAX-1, 2))+1
    samples[0, 0:maxcol] = 1

    # Initialize the remaining rows of V.
    for idx in range(1, dim):

        # The bits of the integer POLY(I) gives the form of polynomial:
        degree = int(math.log(POLY[idx], 2))

        #Expand this bit pattern to separate components:
        includ = numpy.array([val == "1" for val in bin(POLY[idx])[-degree:]])

        #Calculate the remaining elements of row I as explained
        #in Bratley and Fox, section 2.
        for idy in range(degree+1, maxcol+1):
            newv = samples[idx, idy-degree-1].item()
            base = 1
            for idz in range(1, degree+1):
                base *= 2
                if includ[idz-1]:
                    newv = newv ^ base * samples[idx, idy-idz-1].item()
            samples[idx, idy-1] = newv

    samples = samples[:dim]

    # Multiply columns of V by appropriate power of 2.
    samples *= 2**(numpy.arange(maxcol, 0, -1, dtype=int))

    #RECIPD is 1/(common denominator of the elements in V).
    recipd = 0.5**(maxcol+1)
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
