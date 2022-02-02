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

Chaospy is a numerical toolbox for performing uncertainty quantification using
polynomial chaos expansions, advanced Monte Carlo methods implemented in
Python. It also include a full suite of tools for doing low-discrepancy
sampling, quadrature creation, polynomial manipulations, and a lot more.

The philosophy behind ``chaospy`` is not to be a single tool that solves every
uncertainty quantification problem, but instead be a specific tools to aid to
let the user solve problems themselves. This includes both well established
problems, but also to be a foundry for experimenting with new problems, that
are not so well established. To do this, emphasis is put on the following:

* Focus on an easy to use interface that embraces the `pythonic code style
  <https://docs.python-guide.org/writing/style/>`_.
* Make sure the code is "composable", such a way that changing one part of the
  code with something user defined should be easy and encouraged.
* Try to support a broad width of the various methods for doing uncertainty
  quantification where that makes sense to involve ``chaospy``.
* Make sure that ``chaospy`` plays nice with a large set of of other other
  similar projects. This includes `numpy <https://numpy.org/>`_, `scipy
  <https://scipy.org/>`_, `scikit-learn <https://scikit-learn.org>`_,
  `statsmodels <https://statsmodels.org/>`_, `openturns
  <https://openturns.org/>`_, and `gstools <https://geostat-framework.org/>`_
  to mention a few.
* Contribute all code to the community open source.

Installation
============

Installation should be straight forward from `pip <https://pypi.org/>`_:

.. code-block:: bash

    pip install chaospy

Or if `Conda <https://conda.io/>`_ is more to your liking:

.. code-block:: bash

    conda install -c conda-forge chaospy

Then go over to the `documentation <https://chaospy.readthedocs.io/en/master>`_
to see how to use the toolbox.

Development
===========

Installing ``chaospy`` and its dependencies in developer mode is done as
follows:

.. code-block:: bash

    pip install -r requirements-dev.txt
    pip install -e .

Testing
-------

To ensure that the code run on your local system, run the following:

.. code-block:: bash

    pytest --doctest-modules chaospy/ tests/ README.rst

Documentation
-------------

The documentation build assumes that ``pandoc`` is installed on your
system and available in your path.

To build documentation locally on your system, use ``make`` from the ``docs/``
folder:

.. code-block:: bash

    cd docs/
    make html

Run ``make`` without argument to get a list of build targets.
The HTML target stores output to the folder ``doc/.build/html``.
