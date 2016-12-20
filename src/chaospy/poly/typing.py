"""
Functions related to the type of the coefficients.
"""
from __future__ import division

import numpy

from chaospy.poly.base import Poly

DATATYPES = [
    Poly,
    float,
    numpy.float16,
    numpy.float32,
    numpy.float64,
    int,
    numpy.int16,
    numpy.int32,
    numpy.int64,
]

def dtyping(*args):
    """
    Find least common denominator dtype.

    Examples:
        >>> str(dtyping(int, float)) in ("<class 'float'>", "<type 'float'>")
        True
        >>> print(dtyping(int, Poly))
        <class 'chaospy.poly.base.Poly'>
    """
    args = list(args)

    for idx, arg in enumerate(args):

        if isinstance(arg, Poly):
            args[idx] = Poly

        elif isinstance(arg, numpy.generic):
            args[idx] = numpy.asarray(arg).dtype

        elif isinstance(arg, (float, int)):
            args[idx] = type(arg)

    for type_ in DATATYPES:
        if type_ in args:
            return type_

    raise ValueError(
        "dtypes not recognised " + str([str(_) for _ in args]))


def asfloat(vari, limit=10**300):
    """
    Convert dtype of polynomial coefficients to float.

    Example:
        >>> poly = 2*cp.variable()+1
        >>> print(poly)
        2q0+1
        >>> print(cp.asfloat(poly))
        2.0q0+1.0
    """
    if isinstance(vari, Poly):
        core = vari.A.copy()
        for key in vari.keys:
            core[key] = core[key]*1.
        return Poly(core, vari.dim, vari.shape, float)

    return numpy.asfarray(vari)


def asint(vari):
    """
    Convert dtype of polynomial coefficients to float.

    Example:
        >>> poly = 1.5*cp.variable()+2.25
        >>> print(poly)
        1.5q0+2.25
        >>> print(cp.asint(poly))
        q0+2
    """
    if isinstance(vari, Poly):

        core = vari.A.copy()
        for key in vari.keys:
            core[key] = numpy.asarray(core[key], dtype=int)

        return Poly(core, vari.dim, vari.shape, int)

    return numpy.asarray(vari, dtype=int)



def tolist(poly):
    """
    Convert polynomial array into a list of polynomials.

    Examples:
        >>> poly = cp.prange(3)
        >>> print(poly)
        [1, q0, q0^2]
        >>> array = cp.tolist(poly)
        >>> print(isinstance(array, list))
        True
        >>> print(array[1])
        q0
    """
    return toarray(poly).tolist()


def toarray(vari):
    """
    Convert polynomial array into a numpy.asarray of polynomials.

    Args:
        vari (Poly, array_like) : Input data.

    Returns:
        Q (numpy.ndarray) : A numpy array with `Q.shape==A.shape`.

    Examples:
        >>> poly = cp.prange(3)
        >>> print(poly)
        [1, q0, q0^2]
        >>> array = cp.toarray(poly)
        >>> print(isinstance(array, numpy.ndarray))
        True
        >>> print(array[1])
        q0
    """
    if isinstance(vari, Poly):
        shape = vari.shape
        out = numpy.asarray(
            [{} for _ in range(numpy.prod(shape))],
            dtype=object
        )
        core = vari.A.copy()
        for key in core.keys():

            core[key] = core[key].flatten()

            for i in range(numpy.prod(shape)):

                if not numpy.all(core[key][i] == 0):
                    out[i][key] = core[key][i]

        for i in range(numpy.prod(shape)):
            out[i] = Poly(out[i], vari.dim, (), vari.dtype)

        out = out.reshape(shape)
        return out

    return numpy.asarray(vari)


import chaospy as cp # pylint: disable=unused-import

if __name__ == '__main__':
    import doctest
    doctest.testmod()
