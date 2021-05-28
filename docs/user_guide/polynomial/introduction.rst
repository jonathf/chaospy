Introduction
============

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
polynomial array, much like :func:`numpy.array` does for numeric in `numpy`_.

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
    [4, 3, -1]
    >>> expon = poly.exponents
    >>> expon
    array([[1, 0],
           [0, 1],
           [0, 0]], dtype=uint32)

Because these three properties uniquely define a polynomial array, they can
also be used to reconstruct the original polynomial:

.. code:: python

    >>> terms = coeff*chaospy.prod(indet**expon, axis=-1)
    >>> terms
    polynomial([4*q0, 3*q1, -1])
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
