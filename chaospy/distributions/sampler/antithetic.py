"""
Create `antithetic variates`_ from variables on the unit hyper-cube.

In statistics, the antithetic variates method is a variance reduction technique
used in Monte Carlo methods. Considering that the error reduction in the
simulated signal (using Monte Carlo methods) has a square root convergence,
a very large number of sample paths is required to obtain an accurate result.
The antithetic variates method reduces the variance of the simulation results.

Antithetic variate can be accessed as a flag ``antithetic`` in the method
``Distribution.sample`` It can either be set to ``True``, for activation, or as an
array of boolean values, which implies it will be used as the flag ``axes`` in
the examples below.

Example usage
-------------

Creating antithetic variates can be done directly from each distribution by
using the ``antithetic`` flag::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> samples = distribution.sample(6, antithetic=True)

Antithetic variates contains compliment values of itself::

    >>> samples.round(4)
    array([0.6536, 0.3464, 0.115 , 0.885 , 0.9503, 0.0497])
    >>> 1-samples.round(4)
    array([0.3464, 0.6536, 0.885 , 0.115 , 0.0497, 0.9503])

Antithetic variates can also be used in multiple dimensions::

    >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    >>> samples = distribution.sample(6, antithetic=True)
    >>> samples.round(4)
    array([[0.0407, 0.9593, 0.0407, 0.9593, 0.3972, 0.6028],
           [0.8417, 0.8417, 0.1583, 0.1583, 0.2071, 0.2071]])
    >>> 1-samples.round(4)
    array([[0.9593, 0.0407, 0.9593, 0.0407, 0.6028, 0.3972],
           [0.1583, 0.1583, 0.8417, 0.8417, 0.7929, 0.7929]])

Lastly, it is also possible to select which axes should be included when
applying the variate by passing a bool array. For axes that are "false", the
value is frozen in place::

    >>> samples = distribution.sample(6, antithetic=[True, False])
    >>> samples.round(4)
    array([[0.3922, 0.6078, 0.1823, 0.8177, 0.7435, 0.2565],
           [0.0696, 0.0696, 0.8853, 0.8853, 0.9526, 0.9526]])
    >>> 1-samples.round(4)
    array([[0.6078, 0.3922, 0.8177, 0.1823, 0.2565, 0.7435],
           [0.9304, 0.9304, 0.1147, 0.1147, 0.0474, 0.0474]])
    >>> samples = distribution.sample(6, antithetic=[False, True])
    >>> samples.round(4)
    array([[0.9311, 0.9311, 0.4154, 0.4154, 0.029 , 0.029 ],
           [0.982 , 0.018 , 0.3396, 0.6604, 0.7067, 0.2933]])
    >>> 1-samples.round(4)
    array([[0.0689, 0.0689, 0.5846, 0.5846, 0.971 , 0.971 ],
           [0.018 , 0.982 , 0.6604, 0.3396, 0.2933, 0.7067]])

.. _antithetic variates: https://en.wikipedia.org/wiki/Antithetic_variates
"""
import numpy


def create_antithetic_variates(samples, axes=()):
    """
    Generate antithetic variables.

    Args:
        samples (numpy.ndarray):
            The samples, assumed to be on the [0, 1]^D hyper-cube, to be
            reflected.
        axes (tuple):
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

    if not len(axes):
        axes = (True,)
    axes = numpy.asarray(axes, dtype=bool).flatten()

    indices = {tuple(axes*idx) for idx in numpy.ndindex((2,)*dims)}
    indices = sorted(indices, reverse=True)
    indices = sorted(indices, key=lambda idx: sum(idx))
    out = [numpy.where(idx, inverse_samples.T, samples.T).T for idx in indices]
    out = numpy.dstack(out).reshape(dims, -1)
    return out
