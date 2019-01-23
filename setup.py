#!/usr/bin/env python
""" Installation script for vectorfields package """

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages
from os import path
from io import open

from vectorfields import __version__ as version

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='vectorfields',
      version=version,
      description='Python module for creating vector fields for use in a game engine, e.g. UnrealEngine4.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Olaf Haag',
      author_email='haag.olaf@gmail.com',
      url='https://github.com/OlafHaag/VecorFields',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 2.7',  # untested
                   'Programming Language :: Python :: 3',
                   'Operating System :: OS Independent',
                   'Environment :: Console',
                   'Development Status :: 3 - Alpha',
                   ],
      keywords='vector fields FGA 3d game engine VFX',
      packages=find_packages(),
      python_requires='>=2.7',
      install_requires=['numpy',
                        'matplotlib'],
      extras_require={'test': ['pytest', 'hypothesis'], },
      project_urls={'Bug Reports': 'https://github.com/OlafHaag/VectorFields/issues',
                    'Source': 'https://github.com/olafhaag/VectorFields/',
                    },
      )
