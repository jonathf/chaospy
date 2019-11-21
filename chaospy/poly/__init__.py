r"""
A simple polynomial can be created through ``variable`` constructor. For
example to construct a simple bivariate polynomial::

    >>> q0, q1 = chaospy.variable(dims=2)
    >>> q0
    polynomial(q0)

A collection of polynomial can be manipulated using basic arithmetic operators
and joined together into polynomial expansions::

    >>> polys = chaospy.polynomial([1, q0, q0*q1])
    >>> polys
    polynomial([1, q0, q0*q1])

Note that constants and simple polynomials can be joined together into arrays
without any problems.

In practice, having the ability to fine tune a polynomial exactly as one
wants it can be useful, but it can also be cumbersome when dealing with
larger arrays for application. To automate the construction of simple
polynomials, there is the ``basis`` constructor. In its simplest forms
it creates an array of simple monomials::

    >>> chaospy.basis(start=4)
    polynomial([1, q0, q0**2, q0**3, q0**4])

It can be expanded to include number of dimensions and a lower bound for the
polynomial order::

    >>> chaospy.basis(start=1, stop=2, dim=2)
    polynomial([q1, q0, q1**2, q0*q1, q0**2])

There is also the possibility to create anisotropic expansions::

    >>> chaospy.basis(start=1, stop=[1, 2])
    polynomial([q1, q0, q1**2, q0*q1, q0*q1**2])

and the possibility to fine tune the sorting of the polynomials::

    >>> chaospy.basis(start=1, stop=2, dim=2, sort="GRI")
    polynomial([q1**2, q0*q1, q0**2, q1, q0])

To manipulate a polynomial there is a collection of tools available. In
table [tab\_poly] they are listed with description. Much of the behavior
is overlapping with the numerical library ``numpy``. The functions that
overlaps in name is backwards compatible, e.g. if anything else than a
polynomial is inserted into ``chaospy.sum``, the argument is passed to
``numpy.sum``.

Any constructed polynomial is a callable. The argument can either be
inserted positional or as keyword arguments ``q0``, ``q1``, ...::

    >>> poly = chaospy.polynomial([1, q0**2, q0*q1])
    >>> poly(2, 3)
    array([1, 4, 6])
    >>> poly(q1=3, q0=2)
    array([1, 4, 6])

The input can be a mix of scalars and arrays, as long as the shapes together
can be joined to gether in a common compatible shape::

    >>> poly(2, [1, 2, 3, 4])
    array([[1, 1, 1, 1],
           [4, 4, 4, 4],
           [2, 4, 6, 8]])

It is also possible to perform partial evaluation, i.e. evaluating some of the
dimensions. This can be done using the keyword arguments::

    >>> poly(q0=2)
    polynomial([1, 4, 2*q1])
    >>> poly(q1=2)
    polynomial([1, q0**2, 2*q0])

The type of return value for the polynomial is ``numpy.ndarray`` if all
dimensions are filled. If not, it returns a new polynomial.

In addition to input scalars, arrays and masked values, it is also possible to
pass simple polynomials as argument. This allows for variable substitution. For
example, to swap two dimensions, one could simply do the following::

    >>> poly(q1, q0)
    polynomial([1, q1**2, q0*q1])

It is also possible to do all the over mentioned methods together in the same
time. For example, partial evaluation and variable substitution::

    >>> poly(q1=q1**3-1)
    polynomial([1, q0**2, -q0+q0*q1**3])
"""
from functools import wraps

from .basis import basis
from .prange import prange
from .variable import variable
from .setdim import setdim

from numpoly import (
    abs, absolute, add, any, all, allclose, around, aspolynomial, atleast_1d,
    atleast_2d, atleast_3d, call, ceil, concatenate, cumsum, decompose, diff,
    divide, dstack, equal, floor, gradient, hessian, hstack, inner, isclose,
    isconstant, isfinite, multiply, ndpoly, negative, not_equal, outer,
    polynomial, positive, power, prod, repeat, rint, round, square, stack,
    subtract, sum, vstack,
)

@wraps(polynomial)
def Poly(*args, **kwargs):
    print("Deprecation warning: "
          "`chaospy.Poly` replaced by `chaospy.polynomial`")
    return polynomial(*args, **kwargs)
