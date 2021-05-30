Polynomial evaluation
---------------------

Polynomials are not polynomials if they can not be evaluated as such. In the
case of ``chaospy``, this can be done using object call. ``chaospy`` supports
calls with both positional arguments and by name. In other words, one argument
per variable:

.. code:: python

    >>> q0, q1 = chaospy.variable(2)
    >>> poly = chaospy.polynomial([1, q0**2, q0*q1])
    >>> poly
    polynomial([1, q0**2, q0*q1])
    >>> poly(2, 1)
    array([1, 4, 2])
    >>> poly(q0=2, q1=1)
    array([1, 4, 2])

Here the return value is a :class:`numpy.ndarray`. However, it is also possible
to get a polynomial in return, given a partial evaluations:

.. code:: python

    >>> poly(3)
    polynomial([1, 9, 3*q1])
    >>> poly(q0=3)
    polynomial([1, 9, 3*q1])

For positional evaluation, to allow for partial evaluations of variables beyond
the first, it is possible to pass a ``None`` value to the polynomial to
indicate that a variable is not to be touched in a partial evaluation. E.g.:

.. code:: python

    >>> poly(None, 2)
    polynomial([1, q0**2, 2*q0])
    >>> poly(q1=2)
    polynomial([1, q0**2, 2*q0])

Vectorized evaluations is also allowed. Just pass any :class:`numpy.ndarray`
compatible object. ``chaospy`` will expand the shape such that it ends up being
``polynomial.shape+input.shape``. For example:

.. code:: python

    >>> poly(q1=range(4))
    polynomial([[1, 1, 1, 1],
                [q0**2, q0**2, q0**2, q0**2],
                [0, q0, 2*q0, 3*q0]])

It is also possible to mix both scalar and vector arguments, as long as they
are broadcastable in `numpy`_ sense. For example:

.. code:: python

    >>> poly(1, [1, 2, 3])
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 2, 3]])
    >>> poly([1, 2, 3], [1, 2, 3])
    array([[1, 1, 1],
           [1, 4, 9],
           [1, 4, 9]])

Passing arguments with an other datatype than the polynomial, results in the
output using the common denomination:

.. code:: python

    >>> poly(0.5)
    polynomial([1.0, 0.25, 0.5*q1])
    >>> poly(q1=1j)
    polynomial([(1+0j), q0**2, 1j*q0])

Assuming the input you want to evaluate is a large matrix and you want an
interface where the matrix is kept intact, you can use :func:`chaospy.call`. E.g.:

.. code:: python

    >>> array = numpy.arange(12).reshape(2, 6)
    >>> array
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11]])
    >>> chaospy.call(poly, array)
    array([[ 1,  1,  1,  1,  1,  1],
           [ 0,  1,  4,  9, 16, 25],
           [ 0,  7, 16, 27, 40, 55]])

Lastly, it is also possible to pass other polynomials as arguments.
This simplifies any form for variable substitution.

.. code:: python

    >>> poly
    polynomial([1, q0**2, q0*q1])
    >>> poly(q0=q1, q1=q0)
    polynomial([1, q1**2, q0*q1])
    >>> poly(None, 1-q1**3)
    polynomial([1, q0**2, -q0*q1**3+q0])

.. _numpy: https://numpy.org/doc/stable
