Polynomials
===========

All polynomials in ``chaospy``, including the expansions used for construction
and the final model approximations, uses
:class:`numpoly.ndpoly <numpoly.baseclass.ndpoly>` polynomial arrays provided
by the `numpoly`_ library. The library's goal is aligned with ``chaospy``'s,
but have extra design goal to be aligned with the `numpy`_ functional
interface. It is therefore in its own repository.

The table of content below consist of external links to the official `numpoly`_
documentation. However, the functions provided are all also available in
``chaospy``. For example, `numpoly`_ provides a
:func:`numpoly.transpose <numpoly.array_function.transpose>`
function, then there are also an ``chaospy.transpose`` that is exactly the
same function.

.. _numpoly: https://github.com/jonathf/numpoly
.. _numpy: https://github.com/numpy/numpy

.. toctree::
    :maxdepth: 1

    Introduction <https://numpoly.readthedocs.io/en/master/introduction.html>
    Initialization <https://numpoly.readthedocs.io/en/master/initialization.html>
    Numpy Wrapper <https://numpoly.readthedocs.io/en/master/array_function.html>
    Polynomial Division <https://numpoly.readthedocs.io/en/master/division.html>
    Comparison Operators <https://numpoly.readthedocs.io/en/master/comparison.html>
    Differentiation <https://numpoly.readthedocs.io/en/master/differentiation.html>
    orthogonality
    lagrange
