.. _polynomials:

Polynomials
===========

.. currentmodule:: chaospy

Variable constructors
---------------------

.. autosummary::
   :toctree: api

   variable
   polynomial
   aspolynomial
   symbols
   polynomial_from_attributes
   ndpoly.from_attributes
   polynomial_from_roots

Expansions
----------

.. autosummary::
   :toctree: api

   monomial
   expansion.lagrange

Orthogonal constructors
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   expansion.stieltjes
   expansion.cholesky
   expansion.gram_schmidt

Pre-defined orthogonal
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   expansion.chebyshev_1
   expansion.chebyshev_2
   expansion.gegenbauer
   expansion.hermite
   expansion.jacobi
   expansion.laguerre
   expansion.legendre

Baseclass
---------

.. autosummary::
   :template: ndpoly.rst
   :toctree: api

   ndpoly

.. autosummary::
   :toctree: api

   ndpoly.coefficients
   ndpoly.dtype
   ndpoly.exponents
   ndpoly.indeterminants
   ndpoly.keys
   ndpoly.names
   ndpoly.values
   ndpoly.KEY_OFFSET

Helper functions
----------------

Leading coefficient
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   lead_coefficient
   lead_exponent
   sortable_proxy

Polynomial specific
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   call
   decompose
   set_dimensions
   isconstant
   ndpoly.isconstant
   ndpoly.todict
   ndpoly.tonumpy

Array creation
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   diag
   diagonal
   full
   full_like
   ones
   ones_like
   zeros
   zeros_like

Arithmetics
~~~~~~~~~~~

.. autosummary::
   :toctree: api

   add
   inner
   matmul
   multiply
   negative
   outer
   positive
   power
   subtract
   square

Division
~~~~~~~~

.. autosummary::
   :toctree: api

   divide
   divmod
   floor_divide
   mod
   poly_divide
   poly_divmod
   poly_remainder
   remainder
   true_divide

Logic
~~~~~

.. autosummary::
   :toctree: api

   any
   all
   allclose
   equal
   greater
   greater_equal
   isclose
   isfinite
   less
   less_equal
   logical_and
   logical_or
   not_equal

Rounding
~~~~~~~~

.. autosummary::
   :toctree: api

   around
   ceil
   floor
   rint
   round
   round_

Sums/Products
~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   cumsum
   mean
   prod
   sum

Differentiation
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   derivative
   diff
   ediff1d
   gradient
   hessian

Min/Max
~~~~~~~

.. autosummary::
   :toctree: api

   amax
   amin
   argmin
   argmax
   max
   maximum
   min
   minimum

Conditionals
~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   choose
   count_nonzero
   nonzero
   where

Save/Load
~~~~~~~~~

.. autosummary::
   :toctree: api

   load
   loadtxt
   save
   savetxt
   savez
   savez_compressed

Stacking/Splitting
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   array_split
   concatenate
   dsplit
   dstack
   hsplit
   hstack
   split
   stack
   vsplit
   vstack

Shape manipulation
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   atleast_1d
   atleast_2d
   atleast_3d
   broadcast_arrays
   expand_dims
   moveaxis
   repeat
   reshape
   tile
   transpose

Miscellaneous
~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   abs
   absolute
   apply_along_axis
   apply_over_axes
   array_repr
   array_str
   common_type
   copyto
   result_type

Global options
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api

   global_options
   get_options
   set_options

Utilities
~~~~~~~~~

.. autosummary::
   :toctree: api

   cross_truncate
   FeatureNotSupported
   glexindex
   glexsort
