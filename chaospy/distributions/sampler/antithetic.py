"""Generate antithetic variables."""
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
