.. _multivariate:

Joint Distributions
-------------------

There are three ways to create a multivariate probability distribution in
``chaospy``: Using the joint constructor
:func:`~chaospy.distributions.operators.joint.J`, the identical independent distribution
constructor: :func:`~chaospy.distribution.operators.iid.Iid`, and to one of the
pre-constructed multivariate distribution defined in :ref:`listdistributions`.

Joint operator ``J``
~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.operators.joint
.. autoclass:: :: chaospy.distributions.operators.joint.J

Independent Identical Distributed ``Iid``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.operators.iid
.. autoclass:: chaospy.distributions.operators.iid.Iid
