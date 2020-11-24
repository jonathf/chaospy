.. image:: https://github.com/jonathf/chaospy/raw/master/docs/_static/chaospy_logo.svg
   :height: 200 px
   :width: 200 px
   :align: center

|circleci| |codecov| |readthedocs| |downloads| |pypi| |binder|

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
.. |binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/jonathf/chaospy/master?filepath=docs%2Ftutorials

Chaospy is a numerical tool for performing uncertainty quantification using
polynomial chaos expansions and advanced Monte Carlo methods implemented in
Python.

* `Documentation <https://chaospy.readthedocs.io/en/master>`_
* `Interactive tutorials with Binder <https://mybinder.org/v2/gh/jonathf/chaospy/master?filepath=docs%2Ftutorials>`_
* `Source code <https://github.com/jonathf/chaospy>`_
* `Issue tracker <https://github.com/jonathf/chaospy/issues>`_
* `Code of Conduct <https://github.com/jonathf/chaospy/blob/master/CODE_OF_CONDUCT.md>`_
* `Contribution Guideline <https://github.com/jonathf/chaospy/blob/master/CONTRIBUTING.md>`_
* `Changelog <https://github.com/jonathf/chaospy/blob/master/CHANGELOg.md>`_

Installation
------------

Installation should be straight forward using `pip <https://pypi.org/>`_:

.. code-block:: bash

    $ pip install chaospy

For more installation details, see the `installation guide
<https://chaospy.readthedocs.io/en/master/installation.html>`_.

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

    >>> distribution = chaospy.J(chaospy.Uniform(1, 2), chaospy.Normal(0, 2))

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

    >>> evaluations = numpy.array([forward_solver(coordinates, sample)
    ...                            for sample in samples.T])

    >>> print(evaluations[:3, :5].round(8))
    [[1.5        1.5        1.5        1.5        1.5       ]
     [1.75       2.00546578 2.29822457 2.63372042 3.0181921 ]
     [1.25       1.09076905 0.95182169 0.83057411 0.72477163]]

Having all these components in place, we have enough components to perform
point collocation. Or in other words, we can create a polynomial approximation
of ``forward_solver``:

.. code-block:: python

    >>> approx_solver = chaospy.fit_regression(expansion, samples, evaluations)

    >>> print(approx_solver[:2].round(4))
    [q0 -0.0002*q0*q1**3+0.0051*q0*q1**2-0.101*q0*q1+q0]

Since the model approximations are polynomials, we can do inference on them
directly. For example:

.. code-block:: python

    >>> expected = chaospy.E(approx_solver, distribution)
    >>> deviation = chaospy.Std(approx_solver, distribution)

    >>> print(expected[:5].round(8))
    [1.5        1.53092356 1.62757217 1.80240142 2.07915608]
    >>> print(deviation[:5].round(8))
    [0.28867513 0.43364958 0.76501802 1.27106355 2.07110879]

For more extensive guides on this approach an others, see the `tutorial
collection`_.

.. _tutorial collection: https://chaospy.readthedocs.io/en/master/tutorials
