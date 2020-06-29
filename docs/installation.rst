.. _installation:

Installation
============

Installation should be straight forward::

    pip install chaospy

And you should be ready to go.

Alternatively, to get the most current experimental version, the code can be
installed from Github as follows::

    git clone git@github.com:jonathf/chaospy.git    # first time only
    cd chaospy/
    git checkout master
    git pull                                        # after the first
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

    poetry run pytest --nbval --doctest-modules \
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

Questions and Contributions
---------------------------

Please feel free to
`file an issue <https://github.com/jonathf/chaospy/issues>`_ for:

* bug reporting
* asking questions related to usage
* requesting new features
* wanting to contribute with code

If you are using this software in work that will be published, please cite the
journal article: `Chaospy: An open source tool for designing methods of
uncertainty quantification <http://dx.doi.org/10.1016/j.jocs.2015.08.008>`_
