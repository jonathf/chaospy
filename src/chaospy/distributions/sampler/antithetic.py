"""
Create `antithetic variates`_ from variables on the unit hyper-cube.

In statistics, the antithetic variates method is a variance reduction technique
used in Monte Carlo methods. Considering that the error reduction in the
simulated signal (using Monte Carlo methods) has a square root convergence,
a very large number of sample paths is required to obtain an accurate result.
The antithetic variates method reduces the variance of the simulation results.

Example usage
-------------

Univariate case::

    >>> print(create_antithetic_variates([0.1, 0.11, 0.88]))
    [[0.1  0.9  0.11 0.89 0.88 0.12]]

Bivariate case::

    >>> print(create_antithetic_variates([[0.11, 0.22], [0.55, 0.66]]))
    [[0.11 0.89 0.11 0.89 0.22 0.78 0.22 0.78]
     [0.55 0.55 0.45 0.45 0.66 0.66 0.34 0.34]]

Freeze axes::

    >>> print(create_antithetic_variates([[0.1, 0.11], [0.88, 0.99]], axes=[1, 0]))
    [[0.1  0.9  0.11 0.89]
     [0.88 0.88 0.99 0.99]]
    >>> print(create_antithetic_variates([[0.1, 0.11], [0.88, 0.99]], axes=[0, 1]))
    [[0.1  0.1  0.11 0.11]
     [0.88 0.12 0.99 0.01]]

Multivariate case::

    >>> print(create_antithetic_variates(
    ...     [[0.1, 0.11], [0.2, 0.22], [0.3, 0.33]], axes=[1, 0, 0]))
    [[0.1  0.9  0.11 0.89]
     [0.2  0.2  0.22 0.22]
     [0.3  0.3  0.33 0.33]]
    >>> print(create_antithetic_variates(
    ...     [[0.1, 0.11], [0.2, 0.22], [0.3, 0.33]], axes=[0, 1, 0]))
    [[0.1  0.1  0.11 0.11]
     [0.2  0.8  0.22 0.78]
     [0.3  0.3  0.33 0.33]]
    >>> print(create_antithetic_variates(
    ...     [[0.1, 0.11], [0.2, 0.22], [0.3, 0.33]], axes=[0, 0, 1]))
    [[0.1  0.1  0.11 0.11]
     [0.2  0.2  0.22 0.22]
     [0.3  0.7  0.33 0.67]]

.. _antithetic variates: https://en.wikipedia.org/wiki/Antithetic_variates>
"""
import numpy


def create_antithetic_variates(samples, axes=None):
    """
    Generate antithetic variables.

    Args:
        samples (array_like):
            The samples, assumed to be on the [0, 1]^D hyper-cube, to be
            reflected.
        axes (array_like, optional):
            Boolean array of which axes to reflect. If This to limit the number
            of points created in higher dimensions by reflecting all axes at
            once.

    Returns (numpy.ndarray):
        Same as ``samples``, but with samples internally reflected. roughly
        equivalent to ``numpy.vstack([samples, 1-samples])`` in one dimensions.
    """
    samples = numpy.asfarray(samples)
    assert numpy.all(samples <= 1) and numpy.all(samples >= 0), (
        "all samples assumed on interval [0, 1].")
    if len(samples.shape) == 1:
        samples = samples.reshape(1, -1)
    inverse_samples = 1-samples
    dims = len(samples)

    if axes is None:
        axes = True
    axes = numpy.array(axes, dtype=bool).flatten()

    indices = {tuple(axes*idx) for idx in numpy.ndindex((2,)*dims)}
    indices = sorted(indices, reverse=True)
    indices = sorted(indices, key=lambda idx: sum(idx))
    out = [numpy.where(idx, inverse_samples.T, samples.T).T for idx in indices]
    out = numpy.dstack(out).reshape(dims, -1)
    return out
