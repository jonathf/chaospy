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

In `numpy`_ a different choice were made. Comparisons of complex numbers are
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
  `numpy`_ arrays:

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

.. _numpy: https://numpy.org/doc/stable
