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

To run test::

    poetry run pytest --nbval --doctest-modules \
        chaospy/ test/ doc/*.rst
    poetry run pytest --nbval --doctest-modules \
        chaospy/ tests/ doc/*.rst doc/*/*.rst \
        doc/tutorials/*.ipynb doc/tutorials/*/*.ipynb

Documentation
-------------

To build documentation locally on your system, use ``make`` from the ``doc/``
folder::

    cd doc/
    make html

Run ``make`` without argument to get a list of build targets. All targets
stores output to the folder ``doc/.build/``.

Questions & Troubleshooting
---------------------------

For any problems and questions you might have related to ``chaospy``, please
feel free to file an `<https://github.com/jonathf/chaospy/issues>`_.
