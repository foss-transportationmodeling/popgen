#!/usr/bin/env python

import os
import sys

import popgen

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
    version=popgen.__version__,
    description='Synthetic Population Generator 2.0',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Karthik Konduri',
    author_email='karthik.charan@gmail.com',
    url='https://github.com/foss-transportationmodeling/popgen',
    packages=[
        'popgen',
    ],
    # package_dir={'popgen': 'popgen'},
    package_data={'popgen': [
        '../tutorials/1_basic_popgen_setup/*.csv',
        '../tutorials/1_basic_popgen_setup/*.yaml']},
    scripts=['bin/popgen_run'],
    include_package_data=True,
    setup_requires=['numpy==1.9.2'],
    install_requires=[
        'PyYAML==3.11',
        'numpy==1.9.2',
        'scipy==0.15.1',
        'pandas==0.16.1'
    ],
    license='Apache License 2.0',
    zip_safe=False,
    keywords='popgen',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Operating System :: OS Independent'
    ],
)
