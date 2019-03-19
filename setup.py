#!/usr/bin/env python
import inspect
import os
import re
from setuptools import setup, find_packages

# move to current directory:
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join("chaospy", "version.py")) as src:
    regex = r"^__version__\s*=\s*['\"]([^'\"]+)['\"]"
    VERSION = re.search(regex, src.read(), flags=re.M).group(1)

with open("README.rst") as src:
    LONG_DESCRIPTION = src.read()

setup(
    name='chaospy',
    version=VERSION,
    url='https://github.com/jonathf/chaospy',
    author="Jonathan Feinberg",
    author_email="jonathf@gmail.com",
    license='MIT',
    platforms='any',
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    ],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
)
