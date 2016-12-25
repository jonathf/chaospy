"""
Ensure arguments in costructor is correct.
"""

import numpy as np

import chaospy.poly


def ensure_shape(core, shape, shape_):
    """Ensure shape is correct."""
    core = core.copy()
    if shape is None:
        shape = shape_
    elif isinstance(shape, int):
        shape = (shape,)

    if tuple(shape) == tuple(shape_):
        return core, shape

    ones = np.ones(shape, dtype=int)
    for key, val in core.items():
        core[key] = val*ones

    return core, shape


def ensure_dtype(core, dtype, dtype_):
    """Ensure dtype is correct."""
    core = core.copy()
    if dtype is None:
        dtype = dtype_

    if dtype_ == dtype:
        return core, dtype

    for key, val in {
            int: chaospy.poly.typing.asint,
            float: chaospy.poly.typing.asfloat,
            np.float32: chaospy.poly.typing.asfloat,
            np.float64: chaospy.poly.typing.asfloat,
    }.items():

        if dtype == key:
            converter = val
            break
    else:
        raise ValueError("dtype not recognised (%s)" % str(dtype))

    for key, val in core.items():
        core[key] = converter(val)
    return core, dtype


def ensure_dim(core, dim, dim_):
    """Ensure that dim is correct."""
    if dim is None:
        dim = dim_
    if not dim:
        return core, 1
    if dim_ == dim:
        return core, int(dim)

    if dim > dim_:
        key_convert = lambda vari: vari[:dim_]
    else:
        key_convert = lambda vari: vari + (0,)*(dim-dim_)

    new_core = {}
    for key, val in core.items():
        key_ = key_convert(key)
        if key_ in new_core:
            new_core[key_] += val
        else:
            new_core[key_] = val

    return new_core, int(dim)
