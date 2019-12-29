.. _polylist:

List of Polynomial Functions
============================

For numpy version >=1.17, the numpy library started to support dispatching
functionality to subclasses. This means that the functions in polynomials with
the same name as a numpy counterpart will work irrespectively if the function
used was from numpy or chaospy.

Numpy Compatible Functions
--------------------------

.. autofunction:: chaospy.poly.abs
.. autofunction:: chaospy.poly.absolute
.. autofunction:: chaospy.poly.add
.. autofunction:: chaospy.poly.any
.. autofunction:: chaospy.poly.all
.. autofunction:: chaospy.poly.allclose
.. autofunction:: chaospy.poly.around
.. autofunction:: chaospy.poly.atleast_1d
.. autofunction:: chaospy.poly.atleast_2d
.. autofunction:: chaospy.poly.atleast_3d
.. autofunction:: chaospy.poly.ceil
.. autofunction:: chaospy.poly.concatenate
.. autofunction:: chaospy.poly.cumsum
.. autofunction:: chaospy.poly.divide
.. autofunction:: chaospy.poly.dstack
.. autofunction:: chaospy.poly.equal
.. autofunction:: chaospy.poly.floor
.. autofunction:: chaospy.poly.hstack
.. autofunction:: chaospy.poly.inner
.. autofunction:: chaospy.poly.isclose
.. autofunction:: chaospy.poly.isfinite
.. autofunction:: chaospy.poly.multiply
.. autofunction:: chaospy.poly.negative
.. autofunction:: chaospy.poly.not_equal
.. autofunction:: chaospy.poly.outer
.. autofunction:: chaospy.poly.positive
.. autofunction:: chaospy.poly.power
.. autofunction:: chaospy.poly.prod
.. autofunction:: chaospy.poly.repeat
.. autofunction:: chaospy.poly.rint
.. autofunction:: chaospy.poly.round
.. autofunction:: chaospy.poly.square
.. autofunction:: chaospy.poly.stack
.. autofunction:: chaospy.poly.subtract
.. autofunction:: chaospy.poly.sum
.. autofunction:: chaospy.poly.vstack

Chaospy Polynomial Functions
----------------------------

.. autofunction:: chaospy.poly.call
.. autofunction:: chaospy.poly.decompose
.. autofunction:: chaospy.poly.diff
.. autofunction:: chaospy.poly.gradient
.. autofunction:: chaospy.poly.hessian
.. autofunction:: chaospy.poly.isconstant
