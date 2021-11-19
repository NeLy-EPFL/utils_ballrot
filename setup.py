from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="utils_ballrot",
    version="0.1",
    packages=["utils_ballrot",],
    author="Florian Aymanns",
    author_email="florian.ayamnns@epfl.ch",
    description="Basic utility functions for processing ball rotation data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/utils_ballrot.git",
    install_requires=["numpy", "matplotlib",],
)
