.. _polynomial:

Polynomial Representation
=========================

Introduction
------------

In `numpy`_ the concept of *array* is generalized to imply arrays of arbitrary
dimension, overlapping with the concept of *scalars*, *matrices* and *tensors*.
To allow arrays of various dimensions to operate together it defines
unambiguous broadcasting rules for what to expect. The results is a library
that is used as the reference for almost all of the numerical Python community.

In mathematical literature the term *polynomial expansions* is used to denote a
collection of polynomials. Though they strictly do not need to, they are often
indexed, giving each polynomial both a label and a position for where to locate
a polynomial relative to the others. Assuming that there always is an index,
one could say that *polynomial expansions* could just as well be termed
*polynomial array*. And using the rules defined in `numpy`_, there no reason
not to also start talking about *multi-dimensional polynomial arrays*.

The main idea here is that in the same way as :class:`numpy.ndarray` are
composed of scalars, :class:`chaospy.ndpoly` -- the baseclass for the
polynomial arrays -- are composed of simpler polynomials. This gives us a
mental model of a polynomial that looks like this:

.. math::

    \Phi(q_1, \dots, q_D) =
        [\Phi_1(q_1, \dots, q_D), \cdots, \Phi_N(q_1, \dots, q_D)]

where :math:`\Phi` is polynomial vector, :math:`N` is the number of terms in
the polynomial sum, and :math:`q_d` is the :math:`d`-th indeterminant name.
This mental model is shown in practice in how chaospy displays its polynomials
in the REPL:

.. code:: python

    >>> q0, q1 = chaospy.variable(2)
    >>> expansion = chaospy.polynomial([1, q0, q1**2])
    >>> expansion
    polynomial([1, q0, q1**2])

Here :func:`chaospy.variable` creates to simple indeterminants, and the
:func:`chaospy.polynomial` constructor joins an array of polynomials into a
polynomial array, much like :func:`numpy.array` does for numerics in `numpy`_.

Another way to look at the polynomials is to keep the polynomial array as a
single polynomial sum: A multivariate polynomial can in the case of ``chaospy``
be defined as:

.. math::

    \Phi(q_1, \dots, q_D) = \sum_{n=1}^N c_n q_1^{k_{1n}} \cdots q_D^{k_{Dn}}

where :math:`c_n` is a multi-dimensional polynomial
coefficients, and :math:`k_{nd}` is the exponent for the :math:`n`-th
polynomial term and the :math:`d`-th indeterminant name.

Neither of the two ways of representing a polynomial array is incorrect, and
serves different purposes. The former works well for visualisation, while the
latter form gives a better mental model of how ``chaospy`` handles its
polynomial internally.

Modelling polynomials by storing the coefficients as multi-dimensional arrays
is deliberate. Assuming few :math:`k_{nd}` and large dimensional :math:`c_n`,
all numerical operations that are limited to the coefficients, can be done
fast, as `numpy`_ can do the heavy lifting.

This way of representing a polynomial also means that to uniquely defined a
polynomial, we only need the three components:

* :attr:`~chaospy.ndpoly.coefficients` -- the polynomial coefficients
  :math:`c_n` as multi-dimensional arrays.
* :attr:`~chaospy.ndpoly.exponents` -- the exponents :math:`k_{nd}` as a
  2-dimensional matrix.
* :attr:`~chaospy.ndpoly.indeterminants` -- the names of the variables,
  typically ``q0``, ``q1``, etc.

We can access these three defining properties directly from any
:class:`chaospy.ndpoly` polynomial. For example, for a simple polynomial with
scalar coefficients:

.. code:: python

    >>> q0, q1 = chaospy.variable(2)
    >>> poly = chaospy.polynomial(4*q0+3*q1-1)
    >>> poly
    polynomial(3*q1+4*q0-1)
    >>> indet = poly.indeterminants
    >>> indet
    polynomial([q0, q1])
    >>> coeff = poly.coefficients
    >>> coeff
    [-1, 4, 3]
    >>> expon = poly.exponents
    >>> expon
    array([[0, 0],
           [1, 0],
           [0, 1]], dtype=uint32)

Because these three properties uniquely define a polynomial array, they can
also be used to reconstruct the original polynomial:

.. code:: python

    >>> terms = coeff*chaospy.prod(indet**expon, axis=-1)
    >>> terms
    polynomial([-1, 4*q0, 3*q1])
    >>> poly = chaospy.sum(terms, axis=0)
    >>> poly
    polynomial(3*q1+4*q0-1)

Here :func:`chaospy.prod` and :func:`chaospy.sum` is used analogous to their
`numpy`_ counterparts :func:`numpy.prod` and :func:`numpy.sum` to multiply and
add terms together over an axis. See :ref:`numpy_functions` for more details on
how this works.

.. note::

    As mentioned the chosen representation works best with relatively few
    :math:`k_{nd}` and large :math:`c_n`. for large number :math:`k_{nd}` and
    relatively small :math:`c_n` however, the advantage disappears. And even
    worse, in the case where polynomial terms :math:`q_1^{k_{1n}} \cdots
    q_D^{k_{Dn}}` are sparsely represented, the ``chaospy`` representation is
    quite memory inefficient. So it is worth keeping in mind that the advantage
    of this implementation depends a little upon what kind of problems you are
    working on. It is not the tool for all problems.

.. _numpy: https://numpy.org/doc/stable

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

Polynomial expansions
---------------------

A simple polynomial can be created through variable constructor
:func:`chaospy.variable`. For example to construct a simple bivariate
polynomial:

.. code:: python

    >>> q0, q1 = chaospy.variable(2)
    >>> q0
    polynomial(q0)

A collection of polynomial can be manipulated using basic arithmetic operators
and joined together into polynomial expansions:

.. code:: python

    >>> poly = chaospy.polynomial([1, q0, 1-q0*q1, q0**2*q1, q0-q1**2])
    >>> poly
    polynomial([1, q0, -q0*q1+1, q0**2*q1, -q1**2+q0])

Note that constants and simple polynomials can be joined together into arrays
without any problems.

In practice, having the ability to fine tune a polynomial exactly as one wants
it can be useful, but it can also be cumbersome when dealing with larger arrays
for application. To automate the construction of simple polynomials, there is
the :func:`chaospy.monomial` constructor. In its simplest forms it creates an
array of simple monomials:

.. code:: python

    >>> chaospy.monomial(5)
    polynomial([1, q0, q0**2, q0**3, q0**4])

It can be expanded to include number of dimensions and a lower bound for the
polynomial order:

.. code:: python

    >>> chaospy.monomial(start=2, stop=3, dimensions=2)
    polynomial([q0**2, q0*q1, q1**2])

Note that the polynomial is here truncated on total order, meaning that the sum
of exponents is limited by the :math:`L_1`-norm.
If a full tensor-product of polynomials, or another norm is wanted in the
truncation, this is also possible using the ``cross_truncation`` flag:

.. code:: python

    >>> chaospy.monomial(2, 3, dimensions=2, cross_truncation=numpy.inf)
    polynomial([q0**2, q0**2*q1, q1**2, q0*q1**2, q0**2*q1**2])
    >>> chaospy.monomial(2, 4, dimensions=2, cross_truncation=0.8)
    polynomial([q0**2, q0**3, q0*q1, q1**2, q1**3])
    >>> chaospy.monomial(2, 4, dimensions=2, cross_truncation=0.0)
    polynomial([q0**2, q0**3, q1**2, q1**3])

Alternative to the :func:`chaospy.monomial` function, it is also possible to
achieve the same expansion using the exponents only. For example:

.. code:: python

    >>> q0**numpy.arange(5)
    polynomial([1, q0, q0**2, q0**3, q0**4])

Or in the multivariate case:

.. code:: python

    >>> q0q1 = chaospy.variable(2)
    >>> expon = [[2, 0], [3, 0], [0, 2], [0, 3]]
    >>> chaospy.prod(q0q1**expon, axis=-1)
    polynomial([q0**2, q0**3, q1**2, q1**3])

To help construct these exponent, there is function :func:`chaospy.glexindex`.
It behave the same as :func:`chaospy.monomial`, but only creates the exponents.
E.g.:

.. code:: python

    >>> chaospy.glexindex(0, 5, 1).T
    array([[0, 1, 2, 3, 4]])
    >>> chaospy.glexindex(2, 3, 2, numpy.inf).T
    array([[2, 2, 0, 1, 2],
           [0, 1, 2, 2, 2]])
    >>> chaospy.glexindex(2, 4, 2, 0.8).T
    array([[2, 3, 1, 0, 0],
           [0, 0, 1, 2, 3]])
    >>> chaospy.glexindex(2, 4, 2, 0.0).T
    array([[2, 3, 0, 0],
           [0, 0, 2, 3]])

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

Comparison operators
--------------------

Because (real) numbers have a natural total ordering, mathematically speaking,
doing comparisons and sorting is at least conceptually for the most part
trivial. There are a few exceptions though. Take for example complex numbers,
which does not have a total ordering, it then there are not always possible to
assess if one number is large than the other. To get around this limitation,
some choices has to be made. For example, in pure Python the choice to raise
exception for all comparison of complex numbers:

.. code:: python

    >>> 1+3j > 3+1j
    Traceback (most recent call last):
      ...
    TypeError: '>' not supported between instances of 'complex' and 'complex'

In ``numpy`` a different choice were made. Comparisons of complex numbers are
supported, but they limit the compare to the real part only, ignoring the
imaginary part of the numbers. For example:

.. code:: python

    >>> (numpy.array([1+1j, 1+3j, 3+1j, 3+3j]) >
    ...  numpy.array([3+3j, 3+1j, 1+3j, 1+1j]))
    array([False, False,  True,  True])

Polynomials does not have a total ordering either, and imposing one requires
many choices dealing with various edge cases. However, it is possible to impose
a total order that is both internally consistent and which is backwards
compatible with the behavior of
``numpy.ndarray``. It requires some design choices, which are opinionated, and
might not always align with everyones taste.

With this in mind, the ordering implemented in ``chaospy`` is defined
as follows:

* Polynomials containing terms with the highest exponents are considered the
  largest:

  .. code:: python

    >>> q0 = chaospy.variable()
    >>> q0 < q0**2 < q0**3
    True

  If the largest polynomial exponent in one polynomial is larger than in
  another, leading coefficients are ignored:

  .. code:: python

    >>> 4*q0 < 3*q0**2 < 2*q0**3
    True

  In the multivariate case, the polynomial order is determined by the sum of
  the exponents across the indeterminants that are multiplied together:

  .. code:: python

    >>> q0, q1 = chaospy.variable(2)
    >>> q0**2*q1**2 < q0*q1**5 < q0**6*q1
    True

  This implies that given a higher polynomial order, indeterminant names are
  ignored:

  .. code:: python

    >>> q0, q1, q2 = chaospy.variable(3)
    >>> q0 < q2**2 < q1**3
    True

  The same goes for any polynomial terms which are not leading:

  .. code:: python

    >>> 4*q0 < q0**2+3*q0 < q0**3+2*q0
    True

  Here leading means the term in the polynomial that is the largest, as
  defined by the rules here so far.

* Polynomials of equal polynomial order are sorted reverse lexicographically:

  .. code:: python

    >>> q0 < q1 < q2
    True

  As with polynomial order, coefficients and lower order terms are also
  ignored:

  .. code:: python

    >>> 4*q0**3+4*q0 < 3*q1**3+3*q1 < 2*q2**3+2*q2
    True

  Composite polynomials of the same order are sorted lexicographically by
  the dominant indeterminant name:

  .. code:: python

    >>> q0**3*q1 < q0**2*q1**2 < q0*q1**3
    True

  If there are more than two indeterminants, the dominant order first
  addresses the first name (sorted lexicographically), then the second, and so
  on:

  .. code:: python

    >>> q0**2*q1**2*q2 < q0**2*q1*q2**2 < q0*q1**2*q2**2
    True

* Polynomials that have the same leading polynomial exponents, are compared by
  the leading polynomial coefficient:

  .. code:: python

    >>> -4*q0 < -1*q0 < 2*q0
    True

  This notion implies that constant polynomials behave in the same way as
  ``numpy`` arrays:

  .. code:: python

    >>> chaospy.polynomial([2, 4, 6]) > 3
    array([False,  True,  True])

* Polynomials with the same leading polynomial and coefficient are compared on
  the next largest leading polynomial:

  .. code:: python

    >>> q0**2+1 < q0**2+2 < q0**2+3
    True

  And if both the first two leading terms are the same, use the third and so
  on:

  .. code:: python

    >>> q0**2+q0+1 < q0**2+q0+2 < q0**2+q0+3
    True

  Unlike for the leading polynomial term, missing terms are considered present
  as 0. E.g.:

  .. code:: python

    >>> q0**2-1 < q0**2 < q0**2+1
    True

These rules together allow for a total comparison for all polynomials.

In ``chaospy``, there are a few global options that can be passed to
:func:`chaospy.set_options` (or :func:`chaospy.global_options`) to change this
behavior. In particular:

``sort_graded``
  Impose that polynomials are sorted by grade, meaning the indices are always
  sorted by the index sum. E.g. ``q0**2*q1**2*q2**2`` has an exponent sum of 6,
  and will therefore be consider larger than both ``q0**3*q1*q2``,
  ``q0*q1**3*q2`` and ``q0*q1*q2**3``. Defaults to true.
``sort_reverse``
  Impose that polynomials are sorted by reverses lexicographical sorting,
  meaning that ``q0*q1**3`` is considered smaller than ``q0**3*q1``, instead of the
  opposite. Defaults to false.

Polynomial division
-------------------

Numerical division can be split into two variants: floor division
(:func:`chaospy.floor_divide`) and true division (:func:`chaospy.true_divide`):

.. code:: python

    >>> dividend = 7
    >>> divisor = 2
    >>> quotient_true = numpy.true_divide(dividend, divisor)
    >>> quotient_true
    3.5
    >>> quotient_floor = numpy.floor_divide(dividend, divisor)
    >>> quotient_floor
    3

The discrepancy between the two can be captured by a remainder
(:func:`chaospy.remainder`), which allow us to more formally define them as
follows:

.. code:: python

    >>> remainder = numpy.remainder(dividend, divisor)
    >>> remainder
    1
    >>> dividend == quotient_floor*divisor+remainder
    True
    >>> dividend == quotient_true*divisor
    True


In the case of polynomials, neither true nor floor division is supported like
this. Instead it support its own kind of polynomial division
:func:`chaospy.poly_divide`. Polynomial division falls back to behave like
floor division for all constants, as it does not round values:

.. code:: python

    >>> q0, q1 = chaospy.variable(2)
    >>> dividend = q0**2+q1
    >>> divisor = q0-1
    >>> quotient = chaospy.poly_divide(dividend, divisor)
    >>> quotient
    polynomial(q0+1.0)

However, like floor division, it can still have remainders using
:func:`chaospy.poly_remainder`. For example:

.. code:: python

    >>> remainder = chaospy.poly_remainder(dividend, divisor)
    >>> remainder
    polynomial(q1+1.0)
    >>> dividend == quotient*divisor+remainder
    True

In ``numpy``, the "Python syntactic sugar" operators have the following
behavior:

* ``/`` is used for true division :func:`chaospy.true_divide`.
* ``//`` is used for floor division :func:`chaospy.floor_divide`.
* ``%`` is used for remainder :func:`chaospy.remainder`.
* ``divmod`` is used for floor division and remainder in combination to save
  computational cost through :func:`chaospy.divmod`.

In ``chaospy``, which takes precedence if any of the values are of
:class:`chaospy.ndpoly` objects, take the following behavior:

* ``/`` is used for polynomial division :func:`chaospy.poly_divide`, which is
  not compatible with `numpy`_.
* ``//`` is still used for floor division :func:`chaospy.floor_divide` which is
  compatible with `numpy`_, and only works if divisor is a constant.
* ``%`` is used for polynomial remainder :func:`chaospy.poly_remainder`, which
  is not backwards compatible.
* ``divmod`` uses :func:`chaospy.poly_divmod` which is used to save computation
  cost by doing :func:`chaospy.poly_divide` and :func:`chaospy.remainder` at
  the same time.
