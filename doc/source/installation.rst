.. _installation:

Installation
============

Installation should be straight forward::

    pip install chaospy


Alternatively, to get the most current version, the code can be installed from
Github as follows::

    git clone git@github.com:jonathf/chaospy.git
    cd chaospy
    pip install -r requirements.txt
    python setupy.py install

The last command might need ``sudo`` prefix, depending on your python setup.

Optional Packages
-----------------

Optionally, to support more regression methods, install the Scikit-learn
package::

    pip install scikit-learn

Testing
-------

To test the build locally::

    pip install -r requirements-dev.txt
    python setup.py test

It will run ``pytest-runner`` and execute all tests.


Questions & Troubleshooting
---------------------------

For any problems and questions you might have related to ``chaospy``, please
feel free to file an `<https://github.com/jonathf/chaospy/issues>`_.
