About Us
========

Code of Conduct
---------------

`Code of Conduct <https://github.com/jonathf/chaospy/blob/master/CODE_OF_CONDUCT.md>`_

Contribution
------------

`Contribution Guideline <https://github.com/jonathf/chaospy/blob/master/CONTRIBUTING.md>`_

Changelog
---------

`Changelog <https://github.com/jonathf/chaospy/blob/master/CHANGELOG.md>`_

Questions
---------

Please feel free to file for the following reasons:

* bug reporting
* asking questions related to usage
* requesting new features
* wanting to contribute with code, examples, tutorials

Citations
---------

If you are going to publish work using this software, then please cite the
article: `Chaospy: An open source tool for designing methods of uncertainty
quantification <http://dx.doi.org/10.1016/j.jocs.2015.08.008>`_

In addition, if you deal with stochastic dependencies, please also cite:
`Multivariate Polynomial Chaos Expansions with Dependent Variables
<https://epubs.siam.org/doi/10.1137/15M1020447>`_.

Related Projects
----------------

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
