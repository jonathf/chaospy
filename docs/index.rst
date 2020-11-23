Chaospy documentation
=====================

.. toctree::
    :hidden:

    get_started
    tutorials/index
    user_guide/index
    reference/index
    about_us

Chaospy is a numerical tool for performing uncertainty quantification using
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

:ref:`installation`
-------------------

Everything need to install ``chaospy`` locally on your own system, bot for normal usage and as a developer that wants to edit the code.

:ref:`tutorial`
---------------

A collection of illustrative tutorials to get inspiration to what you can do
with ``chaospy``. Each example can be looked upon as complete stories
without going into the details too much.

:ref:`user_guide`
-----------------

A more thorough overview of what ``chaospy`` can offer. This section also
provides some of the theoretical background for some of the methods.

:ref:`reference`
----------------

A complete collection of the public API. This includes all functions, classes,
methods which might be of interest when using `chaospy`.

:ref:`about_us`
---------------

Information about who the developers of chaospy are and how to get in contact.
