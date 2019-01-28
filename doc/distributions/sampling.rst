.. _sampling:

Sampling Schemes
----------------

.. automodule:: chaospy.distributions.sampler

Generator function
~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.sampler.generator
.. autofunction:: chaospy.distributions.sampler.generator.generate_samples

Halton Sequences, ``rule="H"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.sampler.sequences.halton
.. autofunction:: chaospy.distributions.sampler.sequences.halton.create_halton_samples

Hammersley Sequence, ``rule="M"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.sampler.sequences.hammersley
.. autofunction:: chaospy.distributions.sampler.sequences.hammersley.create_hammersley_samples

Korobov Lattice, ``rule="K"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.sampler.sequences.korobov
.. autofunction:: chaospy.distributions.sampler.sequences.korobov.create_korobov_samples

Sobol Sequence, ``rule="S"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.sampler.sequences.sobol
.. autofunction:: chaospy.distributions.sampler.sequences.sobol.create_sobol_samples

Latin Hyper-cube Sampling, ``rule="L"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.sampler.latin_hypercube
.. autofunction:: chaospy.distributions.sampler.latin_hypercube.create_latin_hypercube_samples

Regular grid, ``rule="RG"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.sampler.sequences.grid
.. autofunction:: chaospy.distributions.sampler.sequences.grid.create_grid_samples

Chebyshev Sampling, ``rule="C"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: chaospy.distributions.sampler.sequences.chebyshev
.. autofunction:: chaospy.distributions.sampler.sequences.chebyshev.create_chebyshev_samples
