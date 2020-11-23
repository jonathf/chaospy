.. _chaos_expansion:

Fitting chaos expansion
=======================

Point collocation method
------------------------

Point collocation method, or regression based polynomial chaos expansion builds
open the idea of fitting a polynomial chaos expansion to a set of generated
samples and evaluations. The experiment can be done as follows:

* Select a :ref:`distributions`:

  .. code-block:: python

    >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)

  See reference section :ref:`distribution_collection` for an overview of
  available probability distributions.

* Generate :ref:`orthogonality` using :func:`chaospy.generate_expansion`:

  .. code-block:: python

    >>> orthogonal_expansion = chaospy.generate_expansion(2, distribution)
    >>> orthogonal_expansion
    polynomial([1.0, q1, q0, q1**2-1.0, q0*q1, q0**2-1.0])

* Generate samples using :ref:`sampling` (or alternative abscissas from
  :ref:`quadrature`):

  .. code-block:: python

    >>> samples = distribution.sample(
    ...     2*len(orthogonal_expansion), rule="hammersley")
    >>> samples[:, :4]
    array([[ 0.67448975, -1.15034938,  0.31863936, -0.31863936],
           [-1.42607687, -1.02007623, -0.73631592, -0.50240222]])

  See also reference section :ref:`sampling_collection` for an overview of
  available sampling functions.

* Provide a model solver as a function and evaluate the samples using it.

  .. code-block:: python

    >>> def model_solver(param):
    ...     return [param[0]*param[1], param[0]*numpy.e**-param[1]+1]
    >>> solves = numpy.array([model_solver(sample) for sample in samples.T])
    >>> solves[:4].round(8)
    array([[-0.96187423,  3.80745414],
           [ 1.17344406, -2.19038608],
           [-0.23461924,  1.66539168],
           [ 0.16008512,  0.47338898]])

* Bring it all together using :func:`chaospy.fit_regression`:

  .. code-block:: python

    >>> approx_model = chaospy.fit_regression(
    ...      orthogonal_expansion, samples, solves)
    >>> approx_model.round(2)
    polynomial([q0*q1, 0.11*q1**2-1.44*q0*q1+0.05*q0**2-0.09*q1+1.22*q0+0.94])

In this example, the number of collocation points is selected to be twice the
number of unknown coefficients :math:`N+1`. Changing this is obviously
possible. When the number of parameter is equal the number of unknown, the, the
polynomial approximation becomes an interpolation method and overlap with
Lagrange polynomials. If the number of samples are fewer than the number of
unknown, classical least squares can not be used. Instead it possible to use
methods for doing estimation with too few samples.

Pseudo-spectral projection
--------------------------

In practice the following four components are needed to perform pseudo-spectral
projection. (For the "real" spectral projection method, see also `Intrusice
Galerkin tutorial <../tutorials/intrusive_galerkin.ipynb>`_:

* A distribution for the unknown function parameters (as described in
  section :ref:`distributions`). For example:

  .. code-block:: python

      >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)

  See reference section :ref:`distribution_collection` for an overview of
  available probability distributions.

*  Create integration abscissas and weights (as described in :ref:`quadrature`):

  .. code-block:: python

    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     2, distribution, rule="gaussian")
    >>> abscissas.round(2)
    array([[-1.73, -1.73, -1.73,  0.  ,  0.  ,  0.  ,  1.73,  1.73,  1.73],
           [-1.73,  0.  ,  1.73, -1.73,  0.  ,  1.73, -1.73,  0.  ,  1.73]])
    >>> weights.round(3)
    array([0.028, 0.111, 0.028, 0.111, 0.444, 0.111, 0.028, 0.111, 0.028])

  See also reference section :ref:`quadrature_collection` for an overview of
  available sampling functions.

* An orthogonal polynomial expansion (as described in section
  :ref:`orthogonality`) where the weight function is the distribution in the
  first step:

  .. code-block:: python

    >>> expansion = chaospy.generate_expansion(2, distribution)
    >>> expansion
    polynomial([1.0, q1, q0, q1**2-1.0, q0*q1, q0**2-1.0])

* A function evaluated using the nodes generated in the second step.
  For example:

  .. code-block:: python

    >>> def model_solver(q):
    ...     return [q[0]*q[1], q[0]*numpy.e**-q[1]+1]
    >>> solves = numpy.array([model_solver(ab) for ab in abscissas.T])
    >>> solves[:4].round(8)
    array([[ 3.        , -8.7899559 ],
           [-0.        , -0.73205081],
           [-3.        ,  0.69356348],
           [-0.        ,  1.        ]])

* To bring it together, expansion, abscissas, weights and solves are used as
  arguments to create approximation using :func:`chaospy.fit_quadrature`:

  .. code-block:: python

    >>> approx = chaospy.fit_quadrature(
    ...     expansion, abscissas, weights, solves)
    >>> approx.round(4)
    polynomial([q0*q1, -1.5806*q0*q1+1.6382*q0+1.0])

Note that in this case the function output is bivariate. The software is
designed to create an approximation of any discretized model as long as it is
compatible with ``numpy`` shapes.

As mentioned in section :ref:`orthogonality`, moment based construction of
polynomials can be unstable. This might also be the case for the
denominator :math:`\mathbb E{\Phi_n^2}`. So when using three terms
recurrence, it is common to use the recurrence coefficients to estimated
the denominator.

One caveat with using pseudo-spectral projection is that the calculations of
the norms of the polynomials becomes unstable. To mitigate, recurrence
coefficients can be used to calculate them instead with more stability.
To include these stable norms in the calculations, the following change in code
can be added:

.. code-block:: python

    >>> expansion, norms = chaospy.generate_expansion(
    ...     2, distribution, retall=True)
    >>> approx2 = chaospy.fit_quadrature(
    ...     expansion, abscissas, weights, solves, norms=norms)
    >>> approx2.round(4)
    polynomial([q0*q1, -1.5806*q0*q1+1.6382*q0+1.0])

Note that at low polynomial order, the error is very small, so this is not as
big an issue. For example the two approximation are for all intents and
purposes the same:

.. code-block:: python

    >>> chaospy.allclose(approx, approx2)
    True
