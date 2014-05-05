#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from setuptools import setup
import codecs
import os

import version

here = os.path.abspath(os.path.dirname(__file__))

# get version from git tags if possible (and write it to file),
# read version from file otherwise
def get_version(*file_paths):
    try:
        # read version from git tags
        ver = version.read_version_git()
    except:
        # read version from file
        ver = version.read_version_file(here, *file_paths)
    else:
        # write version to file if we got it successfully from git
        version.write_version_file(ver, here, *file_paths)
    return ver

# Get the long description from the relevant file
# Use codecs.open for Python 2 compatibility
with codecs.open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='prx',
    version=get_version('prx', '_version.py'),
    description='Optimization algorithms based on the proximal operator',
    long_description=long_description,

    url='http://github.com/ryanvolz/prx',

    author='Ryan Volz',
    author_email='ryan.volz@gmail.com',

    license='BSD 3-Clause ("BSD New")',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering',
    ],

    keywords='optimization prox proximal convex first-order',

    packages=['prx'],
    install_requires=['Bottleneck', 'numpy', 'scipy'],
)
