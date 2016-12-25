"""
Cunction to combine two dataset together with a tensor product.
"""

import numpy
import chaospy

def combine(args, part=None):
    """
    All linear combination of a list of list.

    Args:
        args (array_like) : List of input arrays.  Components to take linear
            combination of with `args[i].shape=(N[i], M[i])` where N is to be
            taken linear combination of and M is static.  M[i] is set to 1 if
            missing.

    Returns:
        (numpy.array) : matrix of combinations with shape (numpy.prod(N),
            numpy.sum(M)).

    Examples:
        >>> A, B = [1,2], [[4,4],[5,6]]
        >>> print(chaospy.quad.combine([A, B]))
        [[ 1.  4.  4.]
         [ 1.  5.  6.]
         [ 2.  4.  4.]
         [ 2.  5.  6.]]
    """
    args = [cleanup(arg) for arg in args]

    if part is not None:
        parts, orders = part
        if numpy.array(orders).size == 1:
            orders = [int(numpy.array(orders).item())]*len(args)
        parts = numpy.array(parts).flatten()

        for i, arg in enumerate(args):
            m, n = float(parts[i]), float(orders[i])
            l = len(arg)
            args[i] = arg[int(m/n*l):int((m+1)/n*l)]

    shapes = [arg.shape for arg in args]
    size = numpy.prod(shapes, 0)[0]*numpy.sum(shapes, 0)[1]

    if size > 10**9:
        raise MemoryError("Too large sets")

    if len(args) == 1:
        out = args[0]

    elif len(args) == 2:
        out = combine_two(*args)

    else:
        arg1 = combine_two(*args[:2])
        out = combine([arg1,]+args[2:])

    return out


def combine_two(arg1, arg2):
    l1, d1 = arg1.shape
    l2, d2 = arg2.shape
    out = numpy.empty((l1*l2, d1+d2))
    out[:,:d1] = numpy.tile(arg1, l2).reshape(l1*l2, d1)
    out[:,d1:] = numpy.tile(arg2.T, l1).reshape(d2, l1*l2).T
    return out


def cleanup(arg):
    """Clean up the input variable."""
    arg = numpy.asarray(arg)
    if len(arg.shape) <= 1:
        arg = arg.reshape(arg.size, 1)
    elif len(arg.shape) > 2:
        raise ValueError("shapes must be smaller than 3")
    return arg

if __name__=="__main__":
    import doctest
    doctest.testmod()
