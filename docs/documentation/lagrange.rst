.. _lagrange:

Lagrange Polynomials
--------------------

Lagrange polynomials are not a method for creating orthogonal polynomials.
Instead it is an interpolation method for creating an polynomial expansion that
has the property that each polynomial interpolates exactly one point in space
with the value 1 and has the value 0 for all other interpolation values.
For more details, see this `article on Lagrange polynomials`_.

.. autofunction:: chaospy.orthogonal.lagrange.lagrange_polynomial

.. _article on Lagrange polynomials: https://en.wikipedia.org/wiki/Lagrange_polynomial
