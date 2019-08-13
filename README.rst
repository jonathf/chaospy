Chaospy
=======

|circlci| |codecov| |pypi| |readthedocs|

|logo|

.. |circleci| image:: https://circleci.com/gh/jonathf/chaospy/tree/master.svg?style=svg
    :target: https://circleci.com/gh/jonathf/chaospy/tree/master
.. |codecov| image:: https://codecov.io/gh/jonathf/chaospy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jonathf/chaospy
.. |pypi| image:: https://badge.fury.io/py/chaospy.svg
    :target: https://badge.fury.io/py/chaospy
.. |readthedocs| image:: https://readthedocs.org/projects/chaospy/badge/?version=master
    :target: http://chaospy.readthedocs.io/en/master/?badge=master
.. |logo| image:: logo.jpg
    :target: https://gihub.com/jonathf/chaospy

Chaospy is a numerical tool for performing uncertainty quantification using
polynomial chaos expansions and advanced Monte Carlo methods implemented in
Python 2 and 3.

A article in Elsevier Journal of Computational Science has been published
introducing the software: `here
<http://dx.doi.org/10.1016/j.jocs.2015.08.008>`_.  If you are using this
software in work that will be published, please cite this paper.

Installation
------------

Installation should be straight forward::

    pip install chaospy

And you should be ready to go.

Alternatively, to get the most current experimental version, the code can be
installed from Github as follows::

    git clone git@github.com:jonathf/chaospy.git
    cd chaospy
    pip install -r requirements.txt
    python setup.py install

The last command might need ``sudo`` prefix, depending on your python setup.

Optionally, to support more regression methods, install the Scikit-learn
package::

    pip install scikit-learn

Example Usage
-------------

``chaospy`` is created to be simple and modular. A simple script to implement
point collocation method will look as follows:

.. code-block:: python

    import chaospy
    import numpy

    # your code wrapper goes here
    def foo(coord, prm):
        """Function to do uncertainty quantification on."""
        return prm[0] * numpy.e ** (-prm[1] * numpy.linspace(0, 10, 100))

    # bi-variate probability distribution
    distribution = choaspy.J(chaospy.Uniform(1, 2), chaospy.Uniform(0.1, 0.2))

    # polynomial chaos expansion
    polynomial_expansion = chaospy.orth_ttr(8, distribution)

    # samples:
    samples = distribution.sample(1000)

    # evaluations:
    evals = [foo(sample) for sample in samples.T]

    # polynomial approximation
    foo_approx = chaospy.fit_regression(
        polynomial_expansion, samples, evals)

    # statistical metrics
    expected = chaospy.E(foo_approx, distribution)
    deviation = chaospy.Std(foo_approx, distribution)

For a more extensive description of what going on, see the `tutorial
<https://chaospy.readthedocs.io/en/master/tutorial.html>`_.

For a collection of recipes, see the `cookbook
<https://chaospy.readthedocs.io/en/master/cookbook.html>`_.

Related Projects
----------------

Chaospy is being used in other related projects that requires uncertainty
quantification components ``chaospy`` provides.

+-----------------+-----------------------------------------------------------+
| `UncertainPy`_  | Uncertainty quantification and sensitivity analysis,      |
|                 | tailored towards computational neuroscience.              |
+-----------------+-----------------------------------------------------------+
| `STARFiSh`_     | Shell-based, scientific simulation program                |
|                 | for blood flow in mammals.                                |
+-----------------+-----------------------------------------------------------+
| `SurUQ`_        | Surrogate modeling with nested uncertainty quantification.|
+-----------------+-----------------------------------------------------------+

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

.. _UncertainPy: https://github.com/simetenn/uncertainpy
.. _STARFiSh: https://www.ntnu.no/starfish
.. _SurUQ: https://github.com/redmod-team/suruq
.. _OpenTURNS: http://openturns.github.io/openturns/latest
.. _Régis Lebrun: https://github.com/regislebrun
.. _Chaospy Compatibility: http://openturns.github.io/openturns/latest/user_manual/_generated/openturns.ChaospyDistribution.html
.. _OpenTURNS Compatibility: https://chaospy.readthedocs.io/en/master/recipes/external.html#module-chaospy.external.openturns_
.. _orthopy: https://github.com/nschloe/orthopy
.. _quadpy: https://github.com/nschloe/quadpy
.. _Nico Schlömer: https://github.com/nschloe
.. _Florian Künzner: https://github.com/flo2k
.. _sample distribution: https://chaospy.readthedocs.io/en/master/recipes/external.html#module-chaospy.external.samples

Questions & Troubleshooting
---------------------------

For any problems and questions you might have related to ``chaospy``, please
feel free to file an `issue <https://github.com/jonathf/chaospy/issues>`_.
