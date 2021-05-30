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

In `numpy`_, the "Python syntactic sugar" operators have the following
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

.. _numpy: https://numpy.org/doc/stable
