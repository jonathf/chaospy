.. image:: https://raw.github.com/saullocastro/cubature/master/cubature_logo.png
    :align: center

========
Cubature
========

.. contents::

What is Cubature?
-----------------

It is a numerical integration technique.  From
`MathWorld <http://mathworld.wolfram.com/Cubature.html>`_, 
Ueberhuber (1997, p. 71) and Krommer and Ueberhuber 
(1998, pp. 49 and 155-165) use the word "quadrature" to mean numerical
computation of a univariate integral, and "cubature" to mean numerical
computation of a multiple integral.

Python wrapper for the Cubature package
---------------------------------------

From the `Nanostructures and Computation Wiki at MIT
<http://ab-initio.mit.edu/wiki/index.php/Cubature>`_, `Steven W. Johnson
<http://math.mit.edu/~stevenj/>`_ has written a simple C package for
adaptive multidimensional integration (cubature) of vector-valued
functions over hypercubes and this is a
Python wrapper for the referred C package.

Installation
------------

To install in the ``site-packages`` directory and make it importable from
anywhere:

.. code::
   
    python setup.py install

If you are changing the ``_cubature.pyx`` file, you must have Cython
installed in order to create a new ``_cubature.c`` file. The ``setup.py``
script will automatically try to use the Cython compiler first.

If you want to build only a local ``_cubature.pyd`` file, go to
``./cubature`` and type:

.. code::
   
    python setup.py build_ext -i

Running the tests
-----------------

The Python wrapper has been proven using the testing functions
given in ``./cubature/cpackage/test.c``.

To run the full test:

.. code::
  
   import cubature
   cubature.run_test()

A ``test_cubature.txt`` file will be created in the current directory.

The test parameters are:

.. code::

    ndim - integer, number of dimensions to integrate over
    tol - float, error tolerance
    functions - list, with integers from 0 to 7
    maxEval - integer, maximum number of function calls
    fdim - length of the vector returned by the vector-valued integrand

In the full test the following parameters are used:

.. code::

    ndim = 3
    tol = 1.e-5
    functions = [0, 1, 2, 3, 4, 5, 6, 7]
    maxEval = 1000000
    fdim = 5

Alternatively, the user may pass another set of parameters to the test
script, by calling:

.. code::

    cubature.run_test(ndim, tol, functions, maxEval, fdim)

More details about the test procedure are given in the `C Pacakge README
file <https://github.com/saullocastro/cubature/tree/master/cubature/cpackage/README>`_

Examples
--------

Some examples are given in `./examples <https://github.com/saullocastro/cubature/tree/master/examples>`_.

Fork me!
--------

You are welcome to fork this repository and modify it in whatever way you
want. It will also be nice if you could send a push request here in case
you think your modifications is valuable for another person.

License
-------

This wrapper follows the GNU-GPL license terms discribed in the
`C Package <https://github.com/saullocastro/cubature/tree/master/cubature/cpackage/COPYING>`_.
