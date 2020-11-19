.. _installation:

Get Started
===========

Installation should be straight forward from `pip <https://pypi.org/>`_:

.. code-block:: bash

    pip install chaospy

Or if `Conda <https://conda.io/>`_ is more to your liking:

.. code-block:: bash

    conda install -c conda-forge chaospy

Alternatively, to get the most current experimental version, the code can be
installed from `Github <https://github.com/>`_ as follows:

* First time around, download the repository:

  .. code-block:: bash

      git clone git@github.com:jonathf/chaospy.git

* Every time, move into the repository:

  .. code-block:: bash

      cd chaospy/

* After  the first time, you want to update the branch to the most current
  version of ``master``:

  .. code-block:: bash

      git checkout master
      git pull

* Install the latest version of ``chaospy`` with:

  .. code-block:: bash

      pip install .

Development
-----------

Chaospy uses `poetry`_ to manage its development installation. Assuming
`poetry`_ installed on your system, installing ``chaospy`` for development can
be done from the repository root with the command::

    poetry install

This will install all required dependencies and chaospy into a virtual
environment. If you are not already managing your own virtual environment, you
can use poetry to activate and deactivate with::

    poetry shell
    exit

.. _poetry: https://poetry.eustace.io/

Testing
-------

To run test:

.. code-block:: bash

    poetry run pytest --nbval-lax --doctest-modules \
        chaospy/ tests/ docs/*.rst docs/*/*.rst \
        docs/tutorials/*.ipynb docs/tutorials/*/*.ipynb

Documentation
-------------

To build documentation locally on your system, use ``make`` from the ``doc/``
folder:

.. code-block:: bash

    cd doc/
    make html

Run ``make`` without argument to get a list of build targets. All targets
stores output to the folder ``doc/.build/html``.

Note that the documentation build assumes that ``pandoc`` is installed on your
system and available in your path.
