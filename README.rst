.. image:: https://github.com/jonathf/chaospy/raw/master/docs/.static/chaospy_logo.svg
   :height: 200 px
   :width: 200 px
   :align: center

|circleci| |codecov| |readthedocs| |pypi|

|license| |python-versions| |downloads|

.. |circleci| image:: https://img.shields.io/circleci/build/github/jonathf/chaospy/master
    :target: https://circleci.com/gh/jonathf/chaospy/tree/master
.. |codecov| image:: https://img.shields.io/codecov/c/github/jonathf/chaospy
    :target: https://codecov.io/gh/jonathf/chaospy
.. |readthedocs| image:: https://img.shields.io/readthedocs/chaospy
    :target: http://chaospy.readthedocs.io/en/master/?badge=master
.. |pypi| image:: https://img.shields.io/pypi/v/chaospy
    :target: https://pypi.org/project/chaospy
.. |license| image:: https://img.shields.io/pypi/l/chaospy
    :target: https://opensource.org/licenses/MIT
.. |python-versions| image:: https://img.shields.io/pypi/pyversions/chaospy
.. |downloads| image:: https://img.shields.io/pypi/dw/chaospy
    :target: https://pypistats.org/packages/chaospy

Chaospy is a numerical tool for performing uncertainty quantification using
polynomial chaos expansions and advanced Monte Carlo methods implemented in
Python.

.. contents:: Table of Contents:

Installation
------------

Installation should be straight forward::

    pip install chaospy

And you should be ready to go.

Example Usage
-------------

``chaospy`` is created to be simple and modular. A simple script to implement
point collocation method will look as follows:

.. code-block:: python

    import numpy
    import chaospy

Wrap your code in a function:

.. code-block:: python

    coordinates = numpy.linspace(0, 10, 100)
    def foo(coordinates, params):
        """Function to do uncertainty quantification on."""
        param_init, param_rate = params
        return param_init*numpy.e**(-param_rate*coordinates)

Construct a multivariate probability distribution:

.. code-block:: python

    distribution = chaospy.J(chaospy.Uniform(1, 2), chaospy.Uniform(0.1, 0.2))

Construct polynomial chaos expansion:

.. code-block:: python

    polynomial_expansion = chaospy.generate_expansion(8, distribution)

Generate random samples from for example Halton low-discrepancy sequence:

.. code-block:: python

    samples = distribution.sample(1000, rule="halton")

Evaluate function for each sample:

.. code-block:: python

    evals = numpy.array([foo(coordinates, sample) for sample in samples.T])

Bring the parts together using point collocation method:

.. code-block:: python

    foo_approx = chaospy.fit_regression(
        polynomial_expansion, samples, evals)

Derive statistics from model approximation:

.. code-block:: python

    expected = chaospy.E(foo_approx, distribution)
    deviation = chaospy.Std(foo_approx, distribution)
    sobol_main = chaospy.Sens_m(foo_approx, distribution)
    sobol_total = chaospy.Sens_t(foo_approx, distribution)

For a more extensive guides on what is going on, see the `tutorial collection`_.

.. _tutorial collection: https://chaospy.readthedocs.io/en/master/tutorials

Related Projects
----------------

Chaospy is being used in other related projects that requires uncertainty
quantification components ``chaospy`` provides.

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

Also a few shout-outs:

+--------------+--------------------------------------------------------------+
| `OpenTURNS`_ | Thanks to `Régis Lebrun`_ for both proposing a collaboration |
|              | and creating an initial implementation of both               |
|              | `Chaospy Compatibility`_ in `OpenTURNS`_ and                 |
|              | `OpenTURNS Compatibility`_ in ``chaospy``.                   |
+--------------+--------------------------------------------------------------+
| `orthopy`_   | Thanks to `Nico Schlömer`_ for providing the implementation  |
| `quadpy`_    | for several of the quadrature integration methods.           |
+--------------+--------------------------------------------------------------+
| ``UQRF``     | Thanks to `Florian Künzner`_ for providing the               |
|              | implementation for `sample distribution`_.                   |
+--------------+--------------------------------------------------------------+

.. _OpenTURNS: http://openturns.github.io/openturns/latest
.. _Régis Lebrun: https://github.com/regislebrun
.. _Chaospy Compatibility: http://openturns.github.io/openturns/latest/user_manual/_generated/openturns.ChaospyDistribution.html
.. _OpenTURNS Compatibility: https://chaospy.readthedocs.io/en/master/recipes/external.html#module-chaospy.external.openturns_
.. _orthopy: https://github.com/nschloe/orthopy
.. _quadpy: https://github.com/nschloe/quadpy
.. _Nico Schlömer: https://github.com/nschloe
.. _Florian Künzner: https://github.com/flo2k
.. _sample distribution: https://chaospy.readthedocs.io/en/master/recipes/external.html#module-chaospy.external.samples

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
uncertainty quantification <http://dx.doi.org/10.1016/j.jocs.2015.08.008>`_
