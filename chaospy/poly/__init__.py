"""
A simple polynomial can be created through ``variable`` constructor. For
example to construct a simple bivariate polynomial::

     >>> x,y = cp.variable(2)
     >>> print(x)
     q0

A collection of polynomial can be manipulated using basic arithmetic operators
and joined together into polynomial expansions::

     >>> polys = cp.Poly([1, x, x*y])
     >>> print(polys)
     [1, q0, q0q1]

Note that constants and simple polynomials can be joined together into arrays
without any problems.

In practice, having the ability to fine tune a polynomial exactly as one
wants it can be useful, but it can also be cumbersome when dealing with
larger arrays for application. To automate the construction of simple
polynomials, there is the ``basis`` constructor. In its simplest forms
it creates an array of simple monomials::

      >>> print(cp.basis(4))
      [1, q0, q0^2, q0^3, q0^4]

It can be expanded to include number of dimensions and a lower bound for the
polynomial order::

      >>> print(cp.basis(1, 2, dim=2))
      [q0, q1, q0^2, q0q1, q1^2]

There is also the possibility to create anisotropic expansions::

      >>> print(cp.basis(1, [1, 2]))
      [q0, q1, q0q1, q1^2, q0q1^2]

and the possibility to fine tune the sorting of the polynomials::

      >>> print(cp.basis(1, 2, dim=2, sort="GRI"))
      [q0^2, q0q1, q1^2, q0, q1]

To manipulate a polynomial there is a collection of tools available. In
table [tab\_poly] they are listed with description. Much of the behavior
is overlapping with the numerical library ``numpy``. The functions that
overlaps in name is backwards compatible, e.g. if anything else than a
polynomial is inserted into ``cp.sum``, the argument is passed to
``numpy.sum``.

Any constructed polynomial is a callable. The argument can either be
inserted positional or as keyword arguments ``q0``, ``q1``, ...::

      >>> poly = cp.Poly([1, x**2, x*y])
      >>> print(poly(2, 3))
      [1 4 6]
      >>> print(poly(q1=3, q0=2))
      [1 4 6]

The input can be a mix of scalars and arrays, as long as the shapes together
can be joined to gether in a common compatible shape::

      >>> print(poly(2, [1, 2, 3, 4]))
      [[1 1 1 1]
       [4 4 4 4]
       [2 4 6 8]]

It is also possible to perform partial evaluation, i.e. evaluating some of the
dimensions. To tell the polynomial that a dimension should not be evaluated
either leave the argument empty or pass a masked value ``numpy.ma.masked``. For
example::

      >>> print(poly(2))
      [1, 4, 2q1]
      >>> print(poly(np.ma.masked, 2))
      [1, q0^2, 2q0]
      >>> print(poly(q1=2))
      [1, q0^2, 2q0]

The type of return value for the polynomial is ``numpy.ndarray`` if all
dimensions are filled. If not, it returns a new polynomial.

In addition to input scalars, arrays and masked values, it is also possible to
pass simple polynomials as argument. This allows for variable substitution. For
example, to swap two dimensions, one could simply do the following::

      >>> print(poly(y, x))
      [1, q1^2, q0q1]

It is also possible to do all the over mentioned methods together in the same
time. For example, partial evaluation and variable substitution::

      >>> print(poly(q1=y**3-1))
      [1, q0^2, q0q1^3-q0]
"""

import numpy as np
import chaospy as cp

from .base import *
from .collection import *
from .fraction import *
from .wrapper import *
