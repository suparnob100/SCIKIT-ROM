"""Packaging script for the bundled Matplotlib style.

TL;DR
-----
This file defines setup metadata for the small sci_mplstyle_package helper package.

Notes
-----
It includes the publication style file as package data and declares Matplotlib as a dependency.
"""

from setuptools import setup, find_packages

setup(
    name='sci_mplstyle_package',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,  # Ensures that non-Python files are included
    package_data={
        'sci_mplstyle_package': ['style_files/*.mplstyle'],  # Include the .mplstyle file
    },
    install_requires=[
        'matplotlib',  # Ensure matplotlib is installed as a dependency
    ],
    description='A package for custom matplotlib styles',
    author='Suparno Bhattacharyya',
    author_email='suparno.pa@gmail.com',
    url='https://github.com/suparnob100/sci_mplstyle_package',
)
