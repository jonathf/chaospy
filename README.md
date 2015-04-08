# Chaospy
![Logo](logo.jpg)

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


### Prerequisite in Debian/Ubuntu

To install the prerequisite on a Debian/Ubuntu machine:

`sudo apt-get install python-scipy python-networkx cython gcc`

For `scikit-learn`:

`sudo apt-get install build-essential python-dev \`
`       python-setuptools libatlas-dev libatlas3gf-base`

## Installation

To install in the `site-packages` directory and make it importable
from anywhere.

Automated download and installation can be done by running the
following as super user:

`pip install -e git+https://github.com/hplgit/chaospy.git#egg=chaospy`

Alternative, download the Github folder and run the following
command as super user in the root folder:

`python setup.py install`

For `scikit-learn`:

`pip install --user --install-option="--prefix=" -U scikit-learn`

## License

The core code base is licensed under BSD terms.
Files with deviating license have their own license written on top
of the file.
