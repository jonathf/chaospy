.. image:: https://github.com/jonathf/chaospy/raw/master/docs/_static/chaospy_logo.svg
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

* `Documentation <https://chaospy.readthedocs.io/en/master>`_
* `Interactive tutorials with Binder <https://mybinder.org/v2/gh/jonathf/chaospy/master?filepath=docs%2Fuser_guide>`_
* `Code of conduct <https://github.com/jonathf/chaospy/blob/master/CODE_OF_CONDUCT.md>`_
* `Contribution guideline <https://github.com/jonathf/chaospy/blob/master/CONTRIBUTING.md>`_
* `Changelog <https://github.com/jonathf/chaospy/blob/master/CHANGELOG.md>`_
* `License <https://github.com/jonathf/chaospy/blob/master/LICENCE.txt>`_

Chaospy is a numerical toolbox designed for performing uncertainty
quantification through polynomial chaos expansions and advanced Monte
Carlo methods implemented in Python. It includes a comprehensive suite
of tools for low-discrepancy sampling, quadrature creation, polynomial
manipulations, and much more.

The philosophy behind ``chaospy`` is not to serve as a single solution
for all uncertainty quantification challenges, but rather to provide
specific tools that empower users to solve problems themselves. This
approach accommodates well-established problems but also serves as a
foundry for experimenting with new, emerging problems. Emphasis is
placed on the following:

* Focus on an easy-to-use interface that embraces the `pythonic code
  style <https://docs.python-guide.org/writing/style/>`.
* Ensure the code is "composable," meaning it's designed so that users
  can easily and effectively modify parts of the code with their own
  solutions.
* Strive to support a broad range of methods for uncertainty
  quantification where it makes sense to use ``chaospy``.
* Ensure that ``chaospy`` integrates well with a wide array of other
  projects, including `numpy <https://numpy.org/>`, `scipy
  <https://scipy.org/>`, `scikit-learn <https://scikit-learn.org>`,
  `statsmodels <https://statsmodels.org/>`, `openturns
  <https://openturns.org/>`, and `gstools
  <https://geostat-framework.org/>`, among others.
* Contribute all code as open source to the community.

Installation
============

Installation is straightforward via `pip <https://pypi.org/>`_:

.. code-block:: bash

    pip install chaospy

Alternatively, if you prefer `Conda <https://conda.io/>`_:

.. code-block:: bash

    conda install -c conda-forge chaospy

After installation, visit the `documentation
<https://chaospy.readthedocs.io/en/master>`_ to learn how to use the
toolbox.

Development
===========

To install ``chaospy`` and its dependencies in developer mode:

.. code-block:: bash

    pip install -e .[dev]

Testing
-------

To run tests on your local system:

.. code-block:: bash

    pytest --doctest-modules chaospy/ tests/ README.rst

Documentation
-------------

Ensure that ``pandoc`` is installed and available in your path to
build the documentation.

From the ``docs/`` directory, build the documentation locally using:

.. code-block:: bash

    cd docs/
    make html

Run ``make`` without arguments to view other build targets.
The HTML documentation will be output to ``doc/.build/html``.
