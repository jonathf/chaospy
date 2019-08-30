#!/usr/bin/env python
import inspect
import os
import sys
import re
from setuptools import setup, find_packages

# move to current directory:
os.chdir(os.path.dirname(os.path.abspath(__file__)))

VERSION_REGEX = r"^__version__\s*=\s*['\"]([^'\"]+)['\"]"
KWARGS = {}
if sys.version_info == 3:
    KWARGS = {"encoding": "utf8"}


with open(os.path.join("chaospy", "__init__.py"), **KWARGS) as src:
    VERSION = re.search(VERSION_REGEX, src.read(), flags=re.M).group(1)

with open("README.rst", **KWARGS) as src:
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
