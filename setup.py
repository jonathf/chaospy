#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='chaospy',
    version="2.2.0",
    url='https://github.com/jonathf/chaospy',
    author="Jonathan Feinberg",
    author_email="jonathf@gmail.com",
    license='BSD',
    platforms='any',
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy", "scipy", "networkx"
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    ],
)
