import io
import os
import re

from setuptools import find_packages
from setuptools import setup
from os import path


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="tree_nca",
    version="0.1.0",
    url="https://github.com/Thahit/Neural_Cellular_Automata_for_diverse_Tree_growing",
    license='ETH',

    author="Cedric Caspar | TODO",
    author_email="ccaspar@student.ethz.ch",

    description="Adaption of the paper: \"Growing 3D Artefacts and Functional Machines with Neural Cellular Automata\"",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(exclude=('tests',)),

    install_requires=[],

    include_package_data=True,

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
    ],
)
