User-defined Distributions
--------------------------

There are two ways to construct your own custom user-defined probability
distribution in ``chaospy``: Through sub-classing and using a construction
function.

In addition there are one function for using distributions defined from the
`OpenTURNS`_ project.

.. _OpenTURNS: http://openturns.org

Sub-classing ``Dist``
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.baseclass

Distribution Constructor
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.constructor
.. autofunction:: chaospy.distributions.constructor.construct

OpenTURNS Distributions
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: chaospy.distributions.collection.openturns.OTDistribution
