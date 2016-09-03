#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='chaospy',
    version="2.0a",
    url='https://github.com/hplgit/chaospy',
    author="Jonathan Feinberg",
    author_email="jonathf@gmail.com",
    license='BSD',
    platforms='any',
    packages=find_packages(),
    install_requires=[
        "scipy", "networkx"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
    ],
)
