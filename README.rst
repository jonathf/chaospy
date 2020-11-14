.. image:: https://github.com/jonathf/chaospy/raw/master/docs/.static/chaospy_logo.svg
   :height: 200 px
   :width: 200 px
   :align: center

|circleci| |codecov| |readthedocs| |downloads| |pypi|

.. |circleci| image:: https://img.shields.io/circleci/build/github/jonathf/chaospy/master
    :target: https://circleci.com/gh/jonathf/chaospy/tree/master
.. |codecov| image:: https://img.shields.io/codecov/c/github/jonathf/chaospy
    :target: https://codecov.io/gh/jonathf/chaospy
.. |readthedocs| image:: https://img.shields.io/readthedocs/chaospy
    :target: https://chaospy.readthedocs.io/en/master/?badge=master
.. |downloads| image:: https://img.shields.io/pypi/dm/chaospy
    :target: https://pypistats.org/packages/chaospy
.. |pypi| image:: https://img.shields.io/pypi/v/chaospy
    :target: https://pypi.org/project/chaospy

Chaospy is a numerical tool for performing uncertainty quantification using
polynomial chaos expansions and advanced Monte Carlo methods implemented in
Python.

* Documentation: https://chaospy.readthedocs.io/en/master
* Source code: https://github.com/jonathf/chaospy
* Issues: https://github.com/jonathf/chaospy/issues
* Journal article: `"Chaospy: An open source tool for designing methods of
  uncertainty quantification" <http://dx.doi.org/10.1016/j.jocs.2015.08.008>`_

Installation
------------

Installation should be straight forward from `PyPI <https://pypi.org/>`_:

.. code-block:: bash

    $ pip install chaospy

Or alternatively using `Conda <https://>`_:

.. code-block:: bash

    $ conda install -c conda-forge chaospy

Example Usage
-------------

``chaospy`` is created to work well inside numerical Python ecosystem. You
therefore typically need to import `Numpy <https://numpy.org/>`_ along side
``chaospy``:

.. code-block:: python

    >>> import numpy
    >>> import chaospy

``chaospy`` is problem agnostic, so you can use your own code using any means
you find fit. The only requirement is that the output is compatible with
`numpy.ndarray` format:

.. code-block:: python

    >>> coordinates = numpy.linspace(0, 10, 100)
    >>> def forward_solver(coordinates, parameters):
    ...     """Function to do uncertainty quantification on."""
    ...     param_init, param_rate = parameters
    ...     return param_init*numpy.e**(-param_rate*coordinates)

We here assume that ``parameters`` contains aleatory variability with known
probability. We formalize this probability in ``chaospy`` as a joint
probability distribution. For example:

.. code-block:: python

    >>> distribution = chaospy.J(
    ...     chaospy.Uniform(1, 2), chaospy.Normal(0, 2))
    >>> print(distribution)
    J(Uniform(lower=1, upper=2), Normal(mu=0, sigma=2))

Most probability distributions have an associated expansion of orthogonal
polynomials. These can be automatically constructed:

.. code-block:: python

    >>> expansion = chaospy.generate_expansion(8, distribution)
    >>> print(expansion[:5].round(8))
    [1.0 q1 q0-1.5 q0*q1-1.5*q1 q0**2-3.0*q0+2.16666667]

Here the polynomial is defined positional, such that ``q0`` and ``q1`` refers
to the uniform and normal distribution respectively.

The distribution can also be used to create (pseudo-)random samples and
low-discrepancy sequences. For example to create Sobol sequence samples:

.. code-block:: python

    >>> samples = distribution.sample(1000, rule="sobol")
    >>> print(samples[:, :4].round(8))
    [[ 1.5         1.75        1.25        1.375     ]
     [ 0.         -1.3489795   1.3489795  -0.63727873]]

We can evaluating the forward solver using these samples:

.. code-block:: python

    >>> evaluations = numpy.array([
    ...     forward_solver(coordinates, sample) for sample in samples.T])
    >>> print(evaluations[:3, :5].round(8))
    [[1.5        1.5        1.5        1.5        1.5       ]
     [1.75       2.00546578 2.29822457 2.63372042 3.0181921 ]
     [1.25       1.09076905 0.95182169 0.83057411 0.72477163]]

Having all these components in place, we have enough components to perform
point collocation. Or in other words, we can create a polynomial approximation
of ``forward_solver``:

.. code-block:: python

    >>> approx_solver = chaospy.fit_regression(
    ...     expansion, samples, evaluations)
    >>> print(approx_solver[:2].round(4))
    [q0 -0.0002*q0*q1**3+0.0051*q0*q1**2-0.101*q0*q1+q0]

Since the model approximations are polynomials, we can do inference on them
directly. For example:

.. code-block:: python

    >>> expected = chaospy.E(approx_solver, distribution)
    >>> print(expected[:5].round(8))
    [1.5        1.53092356 1.62757217 1.80240142 2.07915608]
    >>> deviation = chaospy.Std(approx_solver, distribution)
    >>> print(deviation[:5].round(8))
    [0.28867513 0.43364958 0.76501802 1.27106355 2.07110879]

For more extensive guides on this approach an others, see the `tutorial
collection`_.

.. _tutorial collection: https://chaospy.readthedocs.io/en/master/tutorials

Contributions and Related Projects
----------------------------------

A few shout-outs to people who have contributed to the Chaospy project:

+--------------+--------------------------------------------------------------+
| `OpenTURNS`_ | Thanks to `Régis Lebrun`_ for both proposing a collaboration |
|              | and creating an initial implementation of both               |
|              | `Chaospy Compatibility`_ in `OpenTURNS`_ and                 |
|              | `OpenTURNS Compatibility`_ in ``chaospy``.                   |
+--------------+--------------------------------------------------------------+
| `orthopy`_   | Thanks to `Nico Schlömer`_ for providing the implementation  |
| `quadpy`_    | for several of the quadrature integration methods.           |
+--------------+--------------------------------------------------------------+
| ``UQRF``     | Thanks to `Florian Künzner`_ for providing the initial       |
|              | implementation of kernel density estimation and              |
|              | quantity-of-interest distribution.                           |
+--------------+--------------------------------------------------------------+

.. _OpenTURNS: http://openturns.github.io/openturns/latest
.. _Régis Lebrun: https://github.com/regislebrun
.. _Chaospy Compatibility: http://openturns.github.io/openturns/latest/user_manual/_generated/openturns.ChaospyDistribution.html
.. _OpenTURNS Compatibility: https://chaospy.readthedocs.io/en/master/recipes/external.html#module-chaospy.external.openturns_
.. _orthopy: https://github.com/nschloe/orthopy
.. _quadpy: https://github.com/nschloe/quadpy
.. _Nico Schlömer: https://github.com/nschloe
.. _Florian Künzner: https://github.com/flo2k

Chaospy is being used in other related projects that requires uncertainty
quantification components ``chaospy`` provides. For example:

+-----------------+-----------------------------------------------------------+
| `easyVVUQ`_     | Library designed to facilitate verification, validation   |
|                 | and uncertainty quantification.                           |
+-----------------+-----------------------------------------------------------+
| `STARFiSh`_     | Shell-based, scientific simulation program                |
|                 | for blood flow in mammals.                                |
+-----------------+-----------------------------------------------------------+
| `Profit`_       | Probabilistic response model fitting via interactive      |
|                 | tools.                                                    |
+-----------------+-----------------------------------------------------------+
| `UncertainPy`_  | Uncertainty quantification and sensitivity analysis,      |
|                 | tailored towards computational neuroscience.              |
+-----------------+-----------------------------------------------------------+
| `SparseSpACE`_  | Spatially adaptive combination technique targeted to      |
|                 | solve high dimensional numerical integration.             |
+-----------------+-----------------------------------------------------------+

.. _easyVVUQ: https://github.com/UCL-CCS/EasyVVUQ
.. _STARFiSh: https://www.ntnu.no/starfish
.. _Profit: https://github.com/redmod-team/profit
.. _UncertainPy: https://github.com/simetenn/uncertainpy
.. _SparseSpACE: https://github.com/obersteiner/sparseSpACE

For a more comprehensive list, see `Github's dependency graph
<https://github.com/jonathf/chaospy/network/dependents>`_.

Questions and Contributions
---------------------------

Please feel free to
`file an issue <https://github.com/jonathf/chaospy/issues>`_ for:

* bug reporting
* asking questions related to usage
* requesting new features
* wanting to contribute with code

If you are using this software in work that will be published, please cite the
journal article: `Chaospy: An open source tool for designing methods of
uncertainty quantification <http://dx.doi.org/10.1016/j.jocs.2015.08.008>`_.

And if you use code to deal with stochastic dependencies, please also cite
`Multivariate Polynomial Chaos Expansions with Dependent Variables
<https://epubs.siam.org/doi/10.1137/15M1020447>`_.
