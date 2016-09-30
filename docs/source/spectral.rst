.. _spectral:

Pseudo-spectral Projection Method
---------------------------------

In practice the following four components are needed to perform psuedo-spectral
projection. (For the "real" spectral projection method, see: :ref:`galerkin`):

-  A distribution for the unknown function parameters as described in
   section :ref:`distributions`. For example::

      >>> distribution = cp.Iid(cp.Normal(0, 1), 2)

-  Create integration absissas and weights::

      >>> absissas, weights = cp.generate_quadrature(
      ...     2, distribution, rule="G")
      >>> print(np.around(absissas, 15))
      [[-1.73205081 -1.73205081 -1.73205081  0.          0.          0.
         1.73205081  1.73205081  1.73205081]
       [-1.73205081  0.          1.73205081 -1.73205081  0.          1.73205081
        -1.73205081  0.          1.73205081]]
      >>> print(weights)
      [ 0.02777778  0.11111111  0.02777778  0.11111111  0.44444444  0.11111111
        0.02777778  0.11111111  0.02777778]

- An orthogonal polynomial expansion as described in section
  :ref:`orthogonality` where the weight function is the distribution in the
  first step::

      >>> orthogonal_expansion = cp.orth_ttr(2, distribution)
      >>> print(orthogonal_expansion)
      [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]

- A function evaluated using the nodes generated in the second step.
  For example::

      >>> def model_solver(q):
      ...     return [q[0]*q[1], q[0]*e**-q[1]+1]
      >>> solves = [model_solver(absissa) for absissa in absissas.T]
      >>> print(solves[:4])

- To bring it together, expansion, absissas, weights and solves are used as
  arguments to create approximation::

      >>> approx = cp.fitter_quad(orth, nodes, weights, solves)
      >>> print(cp.around(approx, 8))
      [q0q1, -1.58058656357q0q1+1.63819248006q0+1.0]

Note that in this case the function output is
bivariate. The software is designed to create an approximation of any
discretized model as long as it is compatible with ``numpy`` shapes.

As mentioned in section :ref:`orthogonality`, moment based construction of
polynomials can be unstable. This might also be the case for the
denominator :math:`\E{\bm\Phi_n^2}`. So when using three terms
recursion, it is common to use the recurrence coefficients to estimated
the denominator.

One cavat with using psuedo-spectral projection is that the calculations of the
norms of the polynomials becomes unstable. To mittigate, recurrence
coefficients can be used to calculate them instead with more stability.
To include these stable norms in the calculations, the following change in code
can be added::

   >>> orthogonal_expansion, norms = cp.orth_ttr(2, dist, retall=True)
   >>> approx2 = cp.fitter_quad(
   ...     orthogonal_expansion, absissas, weights, solves, norms=norms)

Note that at low polynomial order, the error is very small. For example the
largest coefficient between the two approximation::

   >>> print(np.max(abs(approx-approx2).coeffs(), -1))
   [  2.44249065e-15   3.77475828e-15]

The ``coeffs`` function returns all the polynomial coefficients.
