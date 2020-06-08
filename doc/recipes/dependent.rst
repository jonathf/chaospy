.. _dependent:

Stochastic Dependencies
=======================

To make polynomial chaos expansion work in higher dimensions, terms are often
created by multiplying components from the various dimensions together:

.. math::
    \boldsymbol\Phi_{\boldsymbol i} =
      \Phi_{i_1}^{(1)} \Phi_{i_2}^{(2)} \cdots \Phi_{i_D}^{(D)} \qquad
      \boldsymbol i = (i_1, i_2, \dots, i_D)

This allows for the following inner product calculation:

.. math::
     \left\langle
       \boldsymbol\Phi_{\boldsymbol i}, \boldsymbol\Phi_{\boldsymbol j}
     \right\rangle =
     \mathbb E\left[
       \boldsymbol\Phi_{\boldsymbol i}, \boldsymbol\Phi_{\boldsymbol j}
     \right] = \\
     \mathbb E\left[
       \Phi_{i_1}^{(1)}\cdots \Phi_{i_D}^{(D)}
       \Phi_{j_1}^{(1)}\cdots \Phi_{j_D}^{(D)}
     \right] = \\
     \mathbb E\left[
       \Phi_{i_1}^{(1)} \Phi_{j_1}^{(1)}
     \right] \cdots \mathbb E\left[
       \Phi_{i_D}^{(D)} \Phi_{j_D}^{(D)}
     \right] = \\
     \left|
       \Phi_{i_1}^{(1)}
     \right|
     \delta_{i_1 j_1} \cdots
     \left|
       \Phi_{i_D}^{(D)}
     \right|
     \delta_{i_D j_D} =
     \left|
        \Phi_{\boldsymbol{i}}
     \right|
     \delta_{\boldsymbol{i}\boldsymbol{j}}

This allows us to compose multivariate polynomials from univariate from simple
multiplication. For this to work, however, stochastic independences is
typically assumed. Bellow, techniques will show how to deal with stochastically
dependent random variables and polynomial chaos expansions with stochastically
dependent components.

Dependent Random Variables
--------------------------

One of ``chaospy``'s most powerful features is possible to construct advance
multivariate variables directly through dependencies. To illustrate this,
consider the following bivariate distribution::

    >>> dist_ind = chaospy.Gamma(1)
    >>> dist_dep = chaospy.Normal(dist_ind**2, dist_ind+1)
    >>> distribution = chaospy.J(dist_ind, dist_dep)

In other words, a distribution dependent upon another distribution was created
simply by inserting it into the constructor of another distribution. The
resulting bivariate distribution if fully valid with dependent components.
For example the probability density function functions will in this case look
as follows::

    >>> x, y = numpy.meshgrid(numpy.linspace(-4, 7), numpy.linspace(0, 3))
    >>> likelihood = distribution.pdf([x, y])

This method also allows for construct any multivariate probability distribution
as long as you can fully construct the distribution's conditional decomposition
using Rosenblatt transformations. One only has to construct each univariate
probability distribution and add dependencies in through the parameter
structure.

A couple of caveats:

* The underlying feature to accomplish this is using the Rosenblatt
  transformation. Because of this the number of unique random variables in the
  final joint distribution has to be constant. In other words, ``dist_dep`` is
  not a valid distribution in itself, since it is univariate, but depends on
  the results of ``dist_ind``.
* The dependency order does not matter as long as it can defined as an acyclic
  graph. In other words, ``dist_ind`` can not be dependent upon ``dist_dep`` at
  the same time as ``dist_dep`` is dependent upon ``dist_ind``.

Generalized Polynomial Chaos Expansion
--------------------------------------

The most canonical way to deal with stochastically dependent random variables
is to use a technique known as "generalized polynomial chaos expansion" (GPCE).
It is the most popular technique because of its numerical stability compared to
other methods.

To illustrate the technique, let us consider a simple problem where we want to
measure the uncertainty in a model function::

    >>> def u(q):
    ...    x = numpy.linspace(0, 1, 100)
    ...    return q[1]*numpy.exp(-q[0]*x)

where the two parameters ``Q`` are distributed with a multivariate normal
distribution::

    >>> dist_q = chaospy.MvNormal([0.5, 0.5], [[2, 1], [1, 2]])

Under GPCE we select a distribution that is similar as a proxy, which is
uncorrelated. In the case of multivariate normal distribution, this canonically
implies identical independent distributed normal random variable ``R``::

    >>> dist_r = chaospy.Iid(chaospy.Normal(0, 1), 2)

Unlike the dependent ``dist_q``, the independent ``dist_r`` allows for
orthogonal polynomials::

    >>> polynomial = chaospy.orth_ttr(3, dist_r)
    >>> polynomial
    polynomial([1.0, q1, q0, q1**2-1.0, q0*q1, q0**2-1.0, q1**3-3.0*q1,
                q0*q1**2-q0, q0**2*q1-q1, q0**3-3.0*q0])

The idea is that we create our polynomial expansion in ``R`` and link the proxy
variable to the original problem through a map ``T``. These maps can be created
in ``chaospy`` be created as follows::

    >>> forward_map = lambda q: dist_r.inv(dist_q.fwd(q))
    >>> inverse_map = lambda r: dist_q.inv(dist_r.inv(r))

How to employ these maps depends on if point collection or pseudo-spectral
projection is going to be employed. Both will be discussed below.

Point Collocation Method
~~~~~~~~~~~~~~~~~~~~~~~~

In the case of point collocation method, the map can be used directly between
random samples (or pseudo-random sequences)::

    >>> samples_r = dist_r.sample(2*len(polynomial), rule="hammersley")
    >>> samples_q = dist_q.inv(dist_r.fwd(samples_r))

As these samples are linked, they can be used as follows to solve the dependent
problem::

    >>> samples_u = [u(sample) for sample in samples_q.T]
    >>> u_hat = chaospy.fit_regression(polynomial, samples_r, samples_u)

Note here that the ``samples_q`` is used as argument in ``u``, and
``samples_r`` are used in the fit part of the problem.

Pseudo-Spectral Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the case of pseudo-spectral collocation method, both the abscissas and the
weights have to be adjusted. For example::

    >>> abscissas_r, weights_r = chaospy.generate_quadrature(4, dist_r)
    >>> abscissas_q = dist_q.inv(dist_r.fwd(abscissas_r))
    >>> weights_q = weights_r*dist_q.pdf(abscissas_r)/dist_r.pdf(abscissas_q)

These can then be used to solve the dependent problem as follows::

    >>> samples_u = [u(abscissas) for abscissas in abscissas_q.T]
    >>> u_hat = chaospy.fit_quadrature(
    ...   polynomial, abscissas_r, weights_q, samples_u)

Decorrelation Method
--------------------

Unless a orthogonal polynomial expansion is constructed by hand, GPCE is
usually what one wants when addressing stochastic dependencies. However, there
are dependencies where GPCE is a bad match because there are not good mapping.
In such cases, using an alternative, might make more sense.

One such method for dealing with the stochastic dependency beyond GPCE is the
decorrelation method. It is based on the following two observations:

* Any polynomial with expected value 0 is orthogonal to the constant term.
* Orthogonality of two non-constant polynomials are equivalent to the
  polynomials being uncorrelated.

Using this, orthogonality is achieved for a polynomial expansion, by doing the
following:

* Start with any expansion of unique polynomials, correlated or otherwise.
* Temporarily remove the constant term
* Use decorrelation methods, e.g. using Cholesky decomposition to make the
  polynomials mutually uncorrelated.
* Subtract the mean, making the expected value 0 for all polynomials.
* Add the constant term back into the mix.

This method does not assume anything about the dependencies between variables,
only about the dependencies between the polynomial terms. This method is
therefore applicable to dependent variables.

In practice, the decorrelation method using Cholesky decomposition can be done
as follows::

    >>> polynomial = chaospy.orth_chol(3, dist_q)
    >>> samples_q = dist_q.sample(2*len(polynomial), rule="hammersley")
    >>> samples_u = [u(sample) for sample in samples_q.T]
    >>> u_hat = chaospy.fit_regression(polynomial, samples_q, samples_u)

In principle, the same method could be used in pseudo-spectral projection
method. However, to be able to achieve this, the abscissas and weights have to
be tailored to stochastic dependent probability domain. This is outside the
scope of what the ``chaospy`` library is designed to handle.

Gram-Schmidt Orthogonalization Method
-------------------------------------

Gram-Schmidt orthogonalization is a known method for making polynomials
orthogonal. Like the decorrelation method, however, it is known for being
numerically unstable. However, it also does not violate any assumption about
stochastic independence when being used. As such, it can be used as follows::

    >>> polynomial = chaospy.orth_gs(3, dist_q)
    >>> samples_q = dist_q.sample(2*len(polynomial), rule="hammersley")
    >>> samples_u = [u(sample) for sample in samples_q.T]
    >>> u_hat = chaospy.fit_regression(polynomial, samples_q, samples_u)

Same as with the decorrelation method, this method is mostly meant for point
collocation method. However, if one can make a quadrature scheme for the
dependent variables, there isn't any reason for it to not work together with
the Gram-Schidt orthogonal polynomials.
