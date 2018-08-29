# ----------------------------------------------------------------------------
# Copyright (c) 2014, 'prx' developers (see AUTHORS file)
# All rights reserved.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------

# to use a consistent encoding
from codecs import open
from os import path

import versioneer
from setuptools import find_packages, setup

# custom setup.py commands
cmdclass = versioneer.get_cmdclass()

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='prx',
    version=versioneer.get_version(),
    description='Optimization algorithms based on the proximal operator',
    long_description=long_description,

    url='http://github.com/ryanvolz/prx',

    author='Ryan Volz',
    author_email='ryan.volz@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],

    keywords='optimization prox proximal convex first-order',

    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    extras_require={
        'develop': ['flake8', 'nose', 'pylint', 'twine', 'wheel'],
        'doc': ['sphinx'],
    },

    cmdclass=cmdclass,
)
