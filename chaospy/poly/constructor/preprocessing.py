"""
Preprocessing of Poly class arguments.
"""
import numpy as np

import chaospy.poly.constructor


def preprocess(core, dim, shape, dtype):
    """Constructor function for the Poly class."""
    core, dim_, shape_, dtype_ = chaospy.poly.constructor.identify_core(core)

    core, shape = chaospy.poly.constructor.ensure_shape(core, shape, shape_)
    core, dtype = chaospy.poly.constructor.ensure_dtype(core, dtype, dtype_)
    core, dim = chaospy.poly.constructor.ensure_dim(core, dim, dim_)

    # Remove empty elements
    for key in list(core.keys()):
        if np.all(core[key] == 0):
            del core[key]

    assert isinstance(dim, int), \
        "not recognised type for dim: '%s'" % repr(type(dim))
    assert isinstance(shape, tuple), str(shape)
    assert dtype is not None, str(dtype)

    # assert non-empty container
    if not core:
        core = {(0,)*dim: np.zeros(shape, dtype=dtype)}
    else:
        core = {key: np.asarray(value, dtype) for key, value in core.items()}

    return core, dim, shape, dtype
