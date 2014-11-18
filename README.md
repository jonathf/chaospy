
# Chaospy

Chaospy is a numerical tool for performing uncertainty
quantification using polynomial chaos expansions and advanced Monte
Carlo methods.

## Requirements

`python
numpy
scipy
networkx`

### Optional packages

For regression analysis:

`scikit-learn`

For adaptive cubature:

`cython`
`gcc`

### Prerequisite in Debian/Ubuntu

To install the prerequisite on a Debian/Ubuntu machine:

`apt-get install python-numpy python-scipy python-networkx python-sklearn \`

`        cython gcc`

## Installation

To install in the `site-packages` directory and make it importable
from anywhere:

`python setup.py install`

To install the optional Cubature component, go into the subfolder
`cubature` and run the same command there.

## License

The core code base is licensed under BSD terms.
Files with deviating license have their own license written on top
of the file.
