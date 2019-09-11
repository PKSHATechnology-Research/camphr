from setuptools import setup, find_packages
from pip.req import parse_requirements

setup(packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]))
