.. _galerkin:

Intrusive Galerkin Method
=========================

When talking about polynomial chaos expansions, there are typically two
categories methods that are used: non-intrusive and intrusive methods. The
distinction between the two categories lies in how one tries to solve the
problem at hand. In the intrusive methods, the core problem formulation, often
in the form of some governing equations to solve is reformulated to target
a polynomial chaos expansion. In the case of the non-intrusive methods a solver
for deterministic case is used in combination of some form of collocation
method to fit to the expansion.

The ``chaospy`` toolbox caters for the most part to the non-intrusive methods.
With that said, it is still possible to use the toolbox to assist to solve
intrusive formulation.

Problem formulation
-------------------

Consider the following problem that we will solve using intrusive Galerkin
method:

.. math::
   :label: governing

   \frac{d}{dt} u(t) = -a\ u(t) \qquad u(0) = I \qquad t \in [0, 10]

Here :math:`a` and :math:`I` are unknown hyper parameters which can be
described through a probability distribution.

To apply Galerkin's method, we will first assume that the solution :math:`u`
can be expressed as the sum:

.. math::
   :label: expansion

   u(t) = \sum_{n=0}^N c_n(t)\ \Phi_n(a, I)

Here :math:`P_n` are orthogonal polynomials and :math:`c_n` Fourier
coefficients.[#f1]_

Using this, we can substitute :eq:`expansion` in :eq:`governing`:

.. math::

   \frac{d}{dt} \sum_{n=0}^N c_n\ \Phi_n = -a \sum_{n=0}^N c_n \qquad
   \sum_{n=0}^N c_n(0)\ \Phi_n = I


To apply Galerkin's method, we take the inner product of each side of both
equations against the polynomial :math:`\Phi_k` for :math:`k=0,\cdots,N`. For
the first equation, this will have the following form:

.. math::

   \left\langle \frac{d}{dt} \sum_{n=0}^N c_n \Phi_n, \Phi_k \right\rangle &=
   \left\langle -a \sum_{n=0}^N c_n\Phi_n, \Phi_k \right\rangle \\

   \sum_{n=0}^N \frac{d}{dt} c_n \left\langle \Phi_n, \Phi_k \right\rangle &=
   -\sum_{n=0}^N c_n \left\langle a\ \Phi_n, \Phi_n \right\rangle \\

   \frac{d}{dt} c_k \left\langle \Phi_k, \Phi_k \right\rangle &=
   -\sum_{n=0}^N c_n \left\langle a\ \Phi_n, \Phi_k \right\rangle \\

   \frac{d}{dt} c_k &=
   -\sum_{n=0}^N c_n
   \frac{
      \left\langle a\ \Phi_n, \Phi_k \right\rangle
   }{
      \left\langle \Phi_k, \Phi_k \right\rangle
   }

The collapsing of the sum in the third equation is possible because of the
orthogonality property of the polynomials.

The second equation can be formulated as follows:

.. math::

   \left\langle \sum_{n=0}^N c_n(0)\ \Phi_n, \Phi_k \right\rangle &=
   \left\langle I, \Phi_k \right\rangle \\

   \sum_{n=0}^N c_n(0) \left\langle \Phi_n, \Phi_k \right\rangle &=
   \left\langle I, \Phi_k \right\rangle \\

   c_k(0) \left\langle \Phi_k, \Phi_k \right\rangle &=
   \left\langle I, \Phi_k \right\rangle \\

   c_k(0) &=
   \frac{
      \left\langle I, \Phi_k \right\rangle
   }{
      \left\langle \Phi_k, \Phi_k \right\rangle
   }

If we now solve the two set of equations with respect with to :math:`c_k`, then
the problem is solved.

Implementation
--------------

For the save of formality, we start with importing the module we will need in
this example::

   >>> import chaospy as cp
   >>> import numpy as np
   >>> from scipy.integrate import odeint

Here ``odeint`` is a solver for ordinary differential equations, which we will
be using here. For more complicated problems, other solvers have to be applied.

To be able to solve the problem in practice, we start by to defining what the
probability distribution to :math:`a` and :math:`I` should be. We here define
them as follows::

   >>> a = cp.Uniform(0, 0.1)
   >>> I = cp.Uniform(8, 10)
   >>> joint = cp.J(a, I)

With this we can create our orthogonal polynomials. For example for order 5::

   >>> phi, norms = cp.orth_ttr(5, joint, retall=True, normed=True)

To be able to apply ``chaospy`` on the equations, we replace the inner
product formulation with the equivalent expected value formulation:

.. math::

   \frac{d}{dt} c_k &=
   -\sum_{n=0}^N c_n
   \frac{
      \mathbb E\left[ a\ \Phi_n \Phi_k \right]
   }{
      \mathbb E\left[ \Phi_k \Phi_k \right]
   }

   c_k(0) &=
   \frac{
      \mathbb E\left[ I\ \Phi_k \right]
   }{
      \mathbb E\left[ \Phi_k \Phi_k \right]
   }

Solution
--------

Except for the first expectation, all expected values only vary with the index
:math:`k`. We can therefore calculate the following vectors::

   >>> a, I = cp.variable(2)
   >>> expected_pp = cp.E(phi*phi, joint)
   >>> expected_Ip = cp.E(I*phi, joint)

Note that we do not really need ``expected_pp`` since it is equivalent to
``norms`` defined above, which is more numerical stable.

The first expected value varies both along :math:`k` and :math:`n`, so we will
need to formulate it as a matrix::

   >>> phi2 = cp.outer(phi, phi)
   >>> expected_ap2 = cp.E(a*phi2, joint)

From here we must define the right hand side of the first equation which we can
pass to our ODE solver::

   >>> def pend(coeffs, t):
   ...     return -np.sum(coeffs*expected_ap2, -1) / norms

The initial conditions is defined from the second equation::

   >>> cond = expected_Ip / norms

These components are inserted into the ODE solver that returns the
coefficients :math:`c`::

   >>> coeffs = odeint(pend, cond, np.linspace(0, 10, 1000))

The coefficients can then be used to construct the approximation using the
definition in :eq:`expansion`::

   >>> poly_approx = cp.sum(phi*coeffs, -1)

Lastly, this can be used to calculate statistical properties::

   >>> expected_value = cp.E(poly_approx, joint)
   >>> variance = cp.Var(poly_approx, joint)

Alternatively, these values can also be calculated on directly from the
Fourier coefficients.

.. [#f1] Here we could put in the formal definition of orthogonality with weighted function space, but that it outside the scope of this tutorial.
