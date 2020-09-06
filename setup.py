import codecs
import os
from setuptools import setup

from spikes.__version__ import get_versions

name = 'pyspikes'
here = os.path.abspath(os.path.dirname(__file__))


def get_long_description(*parts):
    return codecs.open(os.path.join(here, *parts), 'r').read()


def get_requirements(*parts):
    return codecs.open(os.path.join(here, *parts), 'r').read().splitlines()


setup(
    name=name,
    version=get_versions()['version'],
    packages=['spikes', 'spikes.utils'],
    url='https://github.com/akshaybabloo/Spikes',
    license='BSD-3-Clause',
    author='Akshay Raj Gollahalli',
    author_email='akshay@gollahalli.com',
    description='Spiking Neural Network Spike Encoders',
    long_description=get_long_description('README.md'),
    long_description_content_type="text/markdown",
    install_requires=get_requirements('requirements.txt'),
    keywords="Spiking Neural Network spike encoder",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ]
)
