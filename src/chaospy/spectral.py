r"""
In practice the following four components are needed to perform psuedo-spectral
projection. (For the "real" spectral projection method, see: :ref:`galerkin`):

-  A distribution for the unknown function parameters (as described in
   section :ref:`distributions`). For example::

      >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)

-  Create integration absissas and weights (as described in :ref:`quadrature`)::

    >>> absissas, weights = chaospy.generate_quadrature(
    ...     2, distribution, rule="G")
    >>> print(numpy.around(absissas, 4))
    [[-1.7321 -1.7321 -1.7321  0.      0.      0.      1.7321  1.7321  1.7321]
     [-1.7321  0.      1.7321 -1.7321  0.      1.7321 -1.7321  0.      1.7321]]
    >>> print(numpy.around(weights, 4))
    [0.0278 0.1111 0.0278 0.1111 0.4444 0.1111 0.0278 0.1111 0.0278]

- An orthogonal polynomial expansion (as described in section
  :ref:`orthogonality`) where the weight function is the distribution in the
  first step::

    >>> expansion = chaospy.orth_ttr(2, distribution)
    >>> print(expansion)
    [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]

- A function evaluated using the nodes generated in the second step.
  For example::

    >>> def model_solver(q):
    ...     return [q[0]*q[1], q[0]*numpy.e**-q[1]+1]
    >>> solves = [model_solver(absissa) for absissa in absissas.T]
    >>> print(numpy.around(solves[:4], 8))
    [[ 3.         -8.7899559 ]
     [-0.         -0.73205081]
     [-3.          0.69356348]
     [-0.          1.        ]]

- To bring it together, expansion, absissas, weights and solves are used as
  arguments to create approximation::

    >>> approx = chaospy.fit_quadrature(
    ...     expansion, absissas, weights, solves)
    >>> print(chaospy.around(approx, 8))
    [q0q1, -1.58058656q0q1+1.63819248q0+1.0]

Note that in this case the function output is
bivariate. The software is designed to create an approximation of any
discretized model as long as it is compatible with ``numpy`` shapes.

As mentioned in section :ref:`orthogonality`, moment based construction of
polynomials can be unstable. This might also be the case for the
denominator :math:`\mathbb E{\Phi_n^2}`. So when using three terms
recursion, it is common to use the recurrence coefficients to estimated
the denominator.

One cavat with using psuedo-spectral projection is that the calculations of the
norms of the polynomials becomes unstable. To mittigate, recurrence
coefficients can be used to calculate them instead with more stability.
To include these stable norms in the calculations, the following change in code
can be added::

    >>> expansion, norms = chaospy.orth_ttr(2, distribution, retall=True)
    >>> approx2 = chaospy.fit_quadrature(
    ...     expansion, absissas, weights, solves, norms=norms)
    >>> print(chaospy.around(approx2, 8))
    [q0q1, -1.58058656q0q1+1.63819248q0+1.0]

Note that at low polynomial order, the error is very small. For example the
largest coefficient between the two approximation::

    >>> print(numpy.max(abs(approx-approx2).coeffs(), -1) < 1e-12)
    [ True  True]

The ``coeffs`` function returns all the polynomial coefficients.
"""
import numpy
import chaospy


def fit_quadrature(orth, nodes, weights, solves, retall=False, norms=None, **kws):
    """
    Using spectral projection to create a polynomial approximation over
    distribution space.

    Args:
        orth (Poly):
            Orthogonal polynomial expansion. Must be orthogonal for the
            approximation to be accurate.
        nodes (array_like):
            Where to evaluate the polynomial expansion and model to
            approximate.
            nodes.shape==(D,K) where D is the number of dimensions and K is
            the number of nodes.
        weights (array_like):
            Weights when doing numerical integration.
            weights.shape==(K,)
        solves (array_like, callable):
            The model to approximate.
            If array_like is provided, it must have len(solves)==K.
            If callable, it must take a single argument X with len(X)==D,
            and return a consistent numpy compatible shape.
        norms (array_like):
            In the of TTR using coefficients to estimate the polynomial
            norm is more stable than manual calculation.
            Calculated using quadrature if no provided.
            norms.shape==(len(orth),).

    Returns:
        (chaospy.poly.Poly): fitted model approximation.
    """
    orth = chaospy.poly.Poly(orth)
    nodes = numpy.asfarray(nodes)
    weights = numpy.asfarray(weights)

    if callable(solves):
        solves = [solves(node) for node in nodes.T]
    solves = numpy.asfarray(solves)

    shape = solves.shape
    solves = solves.reshape(weights.size, int(solves.size/weights.size))

    ovals = orth(*nodes)
    vals1 = [(val*solves.T*weights).T for val in ovals]

    if norms is None:
        vals2 = [(val**2*weights).T for val in ovals]
        norms = numpy.sum(vals2, 1)
    else:
        norms = numpy.array(norms).flatten()
        assert len(norms) == len(orth)

    coefs = (numpy.sum(vals1, 1).T/norms).T
    coefs = coefs.reshape(len(coefs), *shape[1:])
    approx_model = chaospy.poly.transpose(chaospy.poly.sum(orth*coefs.T, -1))

    if retall:
        return approx_model, coefs
    return approx_model


if __name__ == "__main__":
    import doctest
    doctest.testmod()
