.. _trucation:

Polynomial Truncation Schemes
=============================

From a theoretical point of view, a spectral projection has infinite polynomial
terms. To make use of a expansion in practice, the number of terms in the
polynomial expansion has to be finite. In the case of polynomial chaos
expansion it is often common to look at the problem as a ordering issue: Which
element goes first, which goes second, and so on? By answering this, which
polynomial terms to include is reduced to two questions:

1. Which ordering scheme should be applied on the polynomial expansion?
2. At which term along the now sorted expansion, should be used as a truncation
   point where all terms there after are ignored?

There are lots of theory that goes into advanced ordering schemes, but
``chaospy`` supports a few variants. For example:

``graded``
    Polynomial of lower polynomial order comes before higher orders. In the
    case of a one dimension monomial, this implies the ordering:
    ``1, x, x^2, x^3, ...``.
``reverse``
    Polynomials letters that show up earlier in the alphabet comes earlier in
    the expansion the letters that come later. For example, in normal ordering
    we have ``z, y, x, w``, while with reverse ordering, it becomes
    ``w, x, y, z``.

The default behavior for all expansion method is sorted *using graded reverse
lexicographical sorting* (grevlex). For example::

    >>> distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Normal(0, 1))
    >>> print(chaospy.orth_ttr(2, distribution))
    [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]

Here the largest polynomial order was provided, and not the number of terms.
This is because this is in most cases what one wants. And if one wants
a specific number of terms it is possible to achieve this through a secondary
manual truncation::

    >>> print(chaospy.orth_ttr(2, distribution)[:4])
    [1.0, q1, q0, q1^2-1.0]

.. _full_tensor_product:

Full Tensor-Product Expansions
------------------------------

In some cases, most often for theoretical consideration, it is convenient to
consider the multivariate polynomial expansion as a tensor-product between
univariate expansion. From such a perspective, the ordering and truncation rule
does not match up the intuitive idea of of the tensor-product. To create the
correct expansion, it is possible to use the flag ``cross_truncation=0`` to
include all terms::

    >>> expansion = chaospy.orth_ttr(2, distribution, cross_truncation=0)
    >>> print(expansion)  # doctest: +NORMALIZE_WHITESPACE
    [1.0, q1, q0, q1^2-1.0, q0q1,
     q0^2-1.0, q0q1^2-q0, q0^2q1-q1, q0^2q1^2-q0^2-q1^2+1.0]

For what it is worth, the same expansion can be created manually with a little
bit of tinkering. First create two univariate expansions to combine::

    >>> expansion1 = chaospy.orth_ttr(2, chaospy.Normal(0, 1))
    >>> expansion2 = chaospy.orth_ttr(2, chaospy.Normal(0, 1))
    >>> print(expansion1, expansion2)
    [1.0, q0, q0^2-1.0] [1.0, q0, q0^2-1.0]

Adjust the dimensions so the dimensions are no longer common::

    >>> expansion2 = chaospy.swapdim(expansion2, dim1=0, dim2=1)
    >>> print(expansion1, expansion2)
    [1.0, q0, q0^2-1.0] [1.0, q1, q1^2-1.0]

From there we create the cross product::

    >>> expansion = chaospy.outer(expansion1, expansion2)
    >>> print(expansion) # doctest: +NORMALIZE_WHITESPACE
    [[1.0,      q1,        q1^2-1.0],
     [q0,       q0q1,      q0q1^2-q0],
     [q0^2-1.0, q0^2q1-q1, q0^2q1^2-q0^2-q1^2+1.0]]

Lastly, we flatten the expansion and put each term in the right order::

    >>> expansion = chaospy.flatten(expansion)
    >>> print(expansion) # doctest: +NORMALIZE_WHITESPACE
    [1.0, q1, q1^2-1.0, q0, q0q1,
     q0q1^2-q0, q0^2-1.0, q0^2q1-q1, q0^2q1^2-q0^2-q1^2+1.0]
    >>> expansion = sorted(expansion, key=lambda q: q.exponents.sum(1).max())
    >>> expansion = chaospy.Poly(expansion)
    >>> print(expansion) # doctest: +NORMALIZE_WHITESPACE
    [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0,
     q0q1^2-q0, q0^2q1-q1, q0^2q1^2-q0^2-q1^2+1.0]

.. _anisotropic_polynomial_expansion:

Anisotropic Polynomial Expansion
--------------------------------

So far all polynomials here have been considered equally. However, in many
application, it is of interest to weight each dimension differently. One
obvious way to do so, is to let the truncation scheme operate differently for
different dimension. For example, consider the following simple bivariate
expansion::

    >>> distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Normal(0, 1))
    >>> print(chaospy.orth_ttr(2, distribution))
    [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]

To let one dimension have greater weight than another, the first positional
argument of ``orth_ttr`` can receive a vector of value indicating the max order
for each dimension. For example::

    >>> print(chaospy.orth_ttr([0, 2], distribution))
    [1.0, q0, q0^2-1.0]
    >>> print(chaospy.orth_ttr([1, 2], distribution))
    [1.0, q1, q0, q0q1, q0^2-1.0]
    >>> print(chaospy.orth_ttr([2, 2], distribution))
    [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]

Cross Truncation Schemes
------------------------

By default, the truncation scheme is pure graded: only polynomial order decides
truncation index. In `full_tensor_product` showed how a full tensor-product
grid could be created by setting the flag ``cross_truncation=0``. However, the
flag can be set to other values as well. The flag value applies the limit:

.. math::

    \left(\sum_{d=1}^D \alpha_d^{1/C}\right)^C <= O

here :math:`D` is the number of dimensions, `C` is the ``cross_truncation``
value and :math:`O` is the polynomial order. If you fill in the value 0 and
1 respectivly for :math:`C`, the two expansions listed so far can be created::

    >>> print(chaospy.orth_ttr(2, distribution, cross_truncation=0))  # doctest: +NORMALIZE_WHITESPACE
    [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0,
     q0q1^2-q0, q0^2q1-q1, q0^2q1^2-q0^2-q1^2+1.0]
    >>> print(chaospy.orth_ttr(2, distribution, cross_truncation=1))
    [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]

Following the formula other truncation schemes can be chosen::

    >>> print(chaospy.orth_ttr(2, distribution, cross_truncation=0.01))
    [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0, q0q1^2-q0, q0^2q1-q1]
    >>> print(chaospy.orth_ttr(2, distribution, cross_truncation=2))
    [1.0, q1, q0, q1^2-1.0, q0^2-1.0]
