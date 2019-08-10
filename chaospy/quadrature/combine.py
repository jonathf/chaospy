"""Function to combine two dataset together with a tensor product."""
import numpy
import chaospy


def combine(args):
    """
    All linear combination of a list of list.

    Args:
        args (numpy.ndarray):
            List of input arrays.  Components to take linear combination of
            with ``args[i].shape == (N[i], M[i])`` where ``N`` is to be taken
            linear combination of and ``M`` is constant.  ``M[i]`` is set to
            1 if missing.

    Returns:
        (numpy.array):
            Matrix of combinations with
            ``shape == (numpy.prod(N), numpy.sum(M))``.

    Examples:
        >>> A, B = [1,2], [[4,4],[5,6]]
        >>> print(chaospy.quadrature.combine([A, B]))
        [[1. 4. 4.]
         [1. 5. 6.]
         [2. 4. 4.]
         [2. 5. 6.]]
    """
    args = [cleanup(arg) for arg in args]
    shapes = [arg.shape for arg in args]
    size = numpy.prod(shapes, 0)[0]*numpy.sum(shapes, 0)[1]

    if size > 10**9:
        raise MemoryError("Too large sets")

    out = args[0]
    for arg in args[1:]:
        out = _combine(out, arg)
    return out


def _combine(arg1, arg2):
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
