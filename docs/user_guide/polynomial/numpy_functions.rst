.. _numpy_functions:

Numpy functions
---------------

The ``chaospy`` concept of arrays is taken from `numpy`_. But it goes a bit
deeper than just inspiration. The base class :class:`chaospy.ndpoly` is a
direct subclass of :class:`numpy.ndarray`:

.. code:: python

    >>> issubclass(chaospy.ndpoly, numpy.ndarray)
    True

The intentions is to have a library that is fast with the respect of the number
of coefficients, as it leverages `numpy`_'s speed where possible.

In addition ``chaospy`` is designed to be behave both as you would expect as a
polynomial, but also, where possible, to behave as a `numpy`_ numerical array.
In practice this means that ``chaospy`` provides a lot functions that also
exists in `numpy`_, which does about the same thing. If one of these
``chaospy`` function is provided with a :class:`numpy.ndarray` object, the
returned values is the same as if provided to the `numpy`_ function with the
same name. For example :func:`chaospy.transpose`:

.. code:: python

    >>> num_array = numpy.array([[1, 2], [3, 4]])
    >>> chaospy.transpose(num_array)
    polynomial([[1, 3],
                [2, 4]])

And this works the other way around as well. If a polynomial is provided to the
`numpy`_ function, it will behave the same way as if it was provided to the
``chaospy`` equivalent. So following the same example, we can use
:func:`numpy.transpose` to transpose :class:`chaospy.ndpoly` polynomials:

.. code:: python

    >>> poly_array = chaospy.polynomial([[1, q0-1], [q1**2, 4]])
    >>> numpy.transpose(poly_array)
    polynomial([[1, q1**2],
                [q0-1, 4]])

Though the overlap in functionality between `numpy`_ and ``chaospy`` is large,
there are still lots of functionality which is specific for each of them.
The most obvious, in the case of ``chaospy`` features not found in `numpy`_ is
the ability to evaluate the polynomials:

.. code:: python

    >>> poly = q1**2-q0
    >>> poly
    polynomial(q1**2-q0)
    >>> poly(4, 4)
    12
    >>> poly(4)
    polynomial(q1**2-4)
    >>> poly([1, 2, 3])
    polynomial([q1**2-1, q1**2-2, q1**2-3])

.. _numpy: https://numpy.org/doc/stable

Function compatibility
~~~~~~~~~~~~~~~~~~~~~~

The numpy library comes with a large number of functions for manipulating
:class:`numpy.ndarray` objects. Many of these functions are supported
``chaospy`` as well.

For numpy version >=1.17, the `numpy`_ library introduced dispatching of its
functions to subclasses. This means that functions in ``chaospy`` with the
same name as a numpy counterpart, it will work the same irrespectively if the
function used was from `numpy`_ or ``chaospy``, as the former will pass any
job to the latter.

For example:

.. code:: python

    >>> poly = chaospy.variable()**numpy.arange(4)
    >>> poly
    polynomial([1, q0, q0**2, q0**3])
    >>> chaospy.sum(poly, keepdims=True)
    polynomial([q0**3+q0**2+q0+1])
    >>> numpy.sum(poly, keepdims=True)
    polynomial([q0**3+q0**2+q0+1])

For earlier versions of `numpy`_, the last line will not work and will instead
raise an error.

In addition, not everything is possible to support, and even within the list of
supported functions, not all use cases can be covered. Bit if such an
unsupported edge case is encountered, an :class:`chaospy.FeatureNotSupported`
error should be raised, so it should be obvious when they happen.

As a developer note, ``chaospy`` aims at being backwards compatible with
`numpy`_ as far as possible when it comes to the functions it provides. This
means that all functions below should as far as possible mirror the behavior
their `numpy`_ counterparts, and for polynomial constant, they should be
identical (except for the object type). Function that provides behavior not
covered by `numpy`_ should be placed elsewhere.

.. _numpy: https://numpy.org/doc/stable
