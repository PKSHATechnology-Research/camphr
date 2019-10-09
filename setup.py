from subprocess import check_call

from setuptools import find_packages, setup
from setuptools.command.develop import develop


class PostDevelop(develop):
    def run(self):
        check_call(["pre-commit", "install"])
        develop.run(self)


setup(packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]))
