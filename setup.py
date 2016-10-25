import codecs
import os
from warnings import warn

from Spikes._version import get_versions

try:
    from setuptools import setup  # noqa, analysis:ignore
except ImportError:
    warn("unable to load setuptools. 'setup.py develop' will not work")
    pass
from distutils.core import setup

name = 'Spikes'
here = os.path.abspath(os.path.dirname(__file__))


def get_long_description(*parts):
    return codecs.open(os.path.join(here, *parts), 'r').read()


def get_requirements(*parts):
    return codecs.open(os.path.join(here, *parts), 'r').read().splitlines()


setup(
    name=name,
    version=get_versions()['version'],
    packages=['Spikes'],
    url='https://akshaybabloo.github.io/Spikes/',
    license='BSD-3-Clause',
    author='Akshay Raj Gollahalli',
    author_email='akshay@gollahalli.com',
    description='Spiking Neural Network Spike Encoders',
    long_description=get_long_description('README.md'),
    install_requires=get_requirements('requirements.txt'),
    keywords="Spiking Neural Network spike encoder",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "License :: OSI Approved"
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering",
    ]
)
