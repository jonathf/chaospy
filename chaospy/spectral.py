r"""
In practice the following four components are needed to perform pseudo-spectral
projection. (For the "real" spectral projection method, see: :ref:`galerkin`):

-  A distribution for the unknown function parameters (as described in
   section :ref:`distributions`). For example::

      >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)

-  Create integration abscissas and weights (as described in :ref:`quadrature`)::

    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     2, distribution, rule="gaussian")
    >>> abscissas.round(2)
    array([[-1.73, -1.73, -1.73,  0.  ,  0.  ,  0.  ,  1.73,  1.73,  1.73],
           [-1.73,  0.  ,  1.73, -1.73,  0.  ,  1.73, -1.73,  0.  ,  1.73]])
    >>> weights.round(3)
    array([0.028, 0.111, 0.028, 0.111, 0.444, 0.111, 0.028, 0.111, 0.028])

- An orthogonal polynomial expansion (as described in section
  :ref:`orthogonality`) where the weight function is the distribution in the
  first step::

    >>> expansion = chaospy.orth_ttr(2, distribution)
    >>> expansion
    polynomial([1.0, q1, q0, q1**2-1.0, q0*q1, q0**2-1.0])

- A function evaluated using the nodes generated in the second step.
  For example::

    >>> def model_solver(q):
    ...     return [q[0]*q[1], q[0]*numpy.e**-q[1]+1]
    >>> solves = numpy.array([model_solver(ab) for ab in abscissas.T])
    >>> solves[:4].round(8)
    array([[ 3.        , -8.7899559 ],
           [-0.        , -0.73205081],
           [-3.        ,  0.69356348],
           [-0.        ,  1.        ]])

- To bring it together, expansion, abscissas, weights and solves are used as
  arguments to create approximation::

    >>> approx = chaospy.fit_quadrature(
    ...     expansion, abscissas, weights, solves)
    >>> approx.round(4)
    polynomial([q0*q1, -1.5806*q0*q1+1.6382*q0+1.0])

Note that in this case the function output is
bivariate. The software is designed to create an approximation of any
discretized model as long as it is compatible with ``numpy`` shapes.

As mentioned in section :ref:`orthogonality`, moment based construction of
polynomials can be unstable. This might also be the case for the
denominator :math:`\mathbb E{\Phi_n^2}`. So when using three terms
recurrence, it is common to use the recurrence coefficients to estimated
the denominator.

One caveat with using pseudo-spectral projection is that the calculations of
the norms of the polynomials becomes unstable. To mitigate, recurrence
coefficients can be used to calculate them instead with more stability.
To include these stable norms in the calculations, the following change in code
can be added::

    >>> expansion, norms = chaospy.orth_ttr(2, distribution, retall=True)
    >>> approx2 = chaospy.fit_quadrature(
    ...     expansion, abscissas, weights, solves, norms=norms)
    >>> approx2.round(4)
    polynomial([q0*q1, -1.5806*q0*q1+1.6382*q0+1.0])

Note that at low polynomial order, the error is very small. For example the
largest coefficient between the two approximation::

    >>> chaospy.allclose(approx, approx2)
    True

The ``coefficients`` function returns all the polynomial coefficients.
"""
import numpy
import numpoly
import chaospy


def fit_quadrature(orth, nodes, weights, solves, retall=False, norms=None, **kws):
    """
    Using spectral projection to create a polynomial approximation over
    distribution space.

    Args:
        orth (numpoly.ndpoly):
            Orthogonal polynomial expansion. Must be orthogonal for the
            approximation to be accurate.
        nodes (numpy.ndarray):
            Where to evaluate the polynomial expansion and model to
            approximate. ``nodes.shape==(D,K)`` where ``D`` is the number of
            dimensions and ``K`` is the number of nodes.
        weights (numpy.ndarray):
            Weights when doing numerical integration. ``weights.shape == (K,)``
            must hold.
        solves (numpy.ndarray):
            The model evaluation to approximate. If `numpy.ndarray` is
            provided, it must have ``len(solves) == K``. If callable, it must
            take a single argument X with ``len(X) == D``, and return
            a consistent numpy compatible shape.
        norms (numpy.ndarray):
            In the of TTR using coefficients to estimate the polynomial norm is
            more stable than manual calculation. Calculated using quadrature if
            no provided. ``norms.shape == (len(orth),)`` must hold.

    Returns:
        (numpoly.ndpoly):
            Fitted model approximation in the form of an polynomial.
    """
    orth = numpoly.polynomial(orth)
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
        norms = numpy.sum(ovals**2*weights, -1)
    else:
        norms = numpy.array(norms).flatten()

    coefs = (numpy.sum(vals1, 1).T/norms).T
    coefs = coefs.reshape(len(coefs), *shape[1:])
    approx_model = numpoly.sum(orth*coefs.T, -1).T

    if retall:
        return approx_model, coefs
    return approx_model
