import numpy as np

import chaospy.poly
from chaospy.poly.base import Poly


def call(poly, args):
    """
    Evaluate a polynomial along specified axes.

    Args:
        poly (Poly) : Input polynomial.
        args (array_like, masked) : Argument to be evalutated.
                Masked values keeps the variable intact.

    Returns:
        (Poly, np.array) : If masked values are used the Poly is returned. Else
                an numpy array matching the polynomial's shape is returned.
    """
    args = list(args)

    # expand args to match dim
    if len(args) < poly.dim:
        args = args + [np.nan]*(poly.dim-len(args))

    elif len(args) > poly.dim:
        raise ValueError("too many arguments")

    # Find and perform substitutions, if any
    x0, x1 = [], []
    for idx, arg in enumerate(args):

        if isinstance(arg, Poly):
            poly_ = Poly({
                tuple(np.eye(poly.dim)[idx]): np.array(1)
            })
            x0.append(poly_)
            x1.append(arg)
            args[idx] = np.nan
    if x0:
        poly = call(poly, args)
        return substitute(poly, x0, x1)

    # Create masks
    masks = np.zeros(len(args), dtype=bool)
    for idx, arg in enumerate(args):
        if np.ma.is_masked(arg) or np.any(np.isnan(arg)):
            masks[idx] = True
            args[idx] = 0

    shape = np.array(
        args[
            np.argmax(
                [np.prod(np.array(arg).shape) for arg in args]
            )
        ]
    ).shape
    args = np.array([np.ones(shape, dtype=int)*arg for arg in args])

    A = {}
    for key in poly.keys:

        key_ = np.array(key)*(1-masks)
        val = np.outer(poly.A[key], np.prod((args.T**key_).T, \
                axis=0))
        val = np.reshape(val, poly.shape + tuple(shape))
        val = np.where(val != val, 0, val)

        mkey = tuple(np.array(key)*(masks))
        if not mkey in A:
            A[mkey] = val
        else:
            A[mkey] = A[mkey] + val

    out = Poly(A, poly.dim, None, None)
    if out.keys and not np.sum(out.keys):
        out = out.A[out.keys[0]]
    elif not out.keys:
        out = np.zeros(out.shape, dtype=out.dtype)
    return out


def substitute(P, x0, x1, V=0):
    """
    Substitute a variable in a polynomial array.

    Args:
        P (Poly) : Input data.
        x0 (Poly, int) : The variable to substitute. Indicated with either unit
                variable, e.g. `x`, `y`, `z`, etc. or through an integer
                matching the unit variables dimension, e.g. `x==0`, `y==1`,
                `z==2`, etc.
        x1 (Poly) : Simple polynomial to substitute `x0` in `P`. If `x1` is an
                polynomial array, an error will be raised.

    Returns:
        (Poly) : The resulting polynomial (array) where `x0` is replaced with
                `x1`.

    Examples:
        >>> x,y = cp.variable(2)
        >>> P = cp.Poly([y*y-1, y*x])
        >>> print(cp.substitute(P, y, x+1))
        [q0^2+2q0, q0^2+q0]

        With multiple substitutions:
        >>> print(cp.substitute(P, [x,y], [y,x]))
        [q0^2-1, q0q1]
    """
    x0,x1 = map(Poly, [x0,x1])
    dim = np.max([p.dim for p in [P,x0,x1]])
    dtype = chaospy.poly.typing.dtyping(P.dtype, x0.dtype, x1.dtype)
    P, x0, x1 = [chaospy.poly.dimension.setdim(p, dim) for p in [P,x0,x1]]

    if x0.shape:
        x0 = [x for x in x0]
    else:
        x0 = [x0]

    if x1.shape:
        x1 = [x for x in x1]
    else:
        x1 = [x1]

    # Check if substitution is needed.
    valid = False
    C = [x.keys[0].index(1) for x in x0]
    for key in P.keys:
        if np.any([key[c] for c in C]):
            valid = True
            break

    if not valid:
        return P

    dims = [tuple(np.array(x.keys[0])!=0).index(True) for x in x0]

    dec = is_decomposed(P)
    if not dec:
        P = decompose(P)

    P = chaospy.poly.dimension.dimsplit(P)

    shape = P.shape
    P = [p for p in chaospy.poly.shaping.flatten(P)]

    for i in range(len(P)):
        for j in range(len(dims)):
            if P[i].keys and P[i].keys[0][dims[j]]:
                P[i] = x1[j].__pow__(P[i].keys[0][dims[j]])
                break

    P = Poly(P, dim, None, dtype)
    P = chaospy.poly.shaping.reshape(P, shape)
    P = chaospy.poly.collection.prod(P, 0)

    if not dec:
        P = chaospy.poly.collection.sum(P, 0)

    return P


def is_decomposed(P):
    """
    Check if a polynomial (array) is on component form.

    Args:
        P (Poly) : Input data.

    Returns:
        (bool) : True if all polynomials in `P` are on component form.

    Examples:
        >>> x,y = cp.variable(2)
        >>> print(cp.is_decomposed(cp.Poly([1,x,x*y])))
        True
        >>> print(cp.is_decomposed(cp.Poly([x+1,x*y])))
        False
    """
    if P.shape:
        return min([is_decomposed(poly) for poly in P])
    return len(P.keys) <= 1


def decompose(P):
    """
    Decompose a polynomial to component form.

    In array missing values are padded with 0 to make decomposition compatible
    with `cp.sum(Q, 0)`.

    Args:
        P (Poly) : Input data.

    Returns:
        (Poly) : Decomposed polynomial with `P.shape==(M,)+Q.shape` where
                `M` is the number of components in `P`.

    Examples:
        >>> q = cp.variable()
        >>> P = cp.Poly([q**2-1, 2])
        >>> print(P)
        [q0^2-1, 2]
        >>> print(cp.decompose(P))
        [[-1, 2], [q0^2, 0]]
        >>> print(cp.sum(cp.decompose(P), 0))
        [q0^2-1, 2]
    """
    P = P.copy()

    if not P:
        return P

    out = [Poly({key:P.A[key]}) for key in P.keys]
    return Poly(out, None, None, None)


import chaospy as cp
