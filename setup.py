#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://popgen.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='popgen',
    version='2.0',
    description='Synthetic Population Generator 2.0',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Karthik Konduri',
    author_email='karthik.charan@gmail.com',
    url='https://github.com/foss-transportationmodeling/popgen',
    packages=[
        'popgen',
    ],
    package_dir={'popgen': 'popgen'},
    include_package_data=True,
    install_requires=[
    ],
    license='Apache License 2.0',
    zip_safe=False,
    keywords='popgen',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
    ],
)
