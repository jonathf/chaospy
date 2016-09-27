Chaospy
=======

|travis| |codecov| |pypi|

|logo|

.. |logo| image:: logo.jpg
.. |travis| image:: https://img.shields.io/travis/jonathf/chaospy.svg
    :target: https://travis-ci.org/jonathf/chaospy
.. |codecov| image:: https://img.shields.io/codecov/c/github/jonathf/chaospy.svg
    :target: https://codecov.io/gh/jonathf/chaospy
.. |pypi| image:: https://img.shields.io/pypi/v/chaospy.svg
    :target: https://pypi.python.org/pypi/chaospy

Chaospy is a numerical tool for performing uncertainty quantification using
polynomial chaos expansions and advanced Monte Carlo methods.

A article in Elsevier Journal of Computational Science has been published
introducing the software:
`here <http://dx.doi.org/10.1016/j.jocs.2015.08.008>`_.
If you are to use this software work that is published, please cite this paper.

Installation
~~~~~~~~~~~~

Installation::

    pip install chaospy

Alternativly, the code can be installed from source as follows::

    pip install -r requirements.txt
    pip install -e .

Optionally, to support more regression methods, install the Scikit-learn
package::

    pip install scikit-learn
