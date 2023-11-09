"""DeepGlow

DeepGlow is a Python package which emulates the BOXFIT gamma-ray-burst afterglow simulation code using a neural network approach.
It can calculate light curves in milliseconds to within a few percent accuracy compared to the original BOXFIT model.

"""

from setuptools import setup

setup(
    name="DeepGlow",
    version="1.1.0",
    description="A neural network emulator for BOXFIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/OMBoersma/DeepGlow",
    author="Oliver Boersma",
    author_email="o.m.boersma@uva.nl",
    license="BSD 2-clause",
    packages=["DeepGlow"],
    install_requires=["tensorflow>=2.0.0", "numpy", "importlib.resources"],
    extras_require={
        "test": ["pytest"],
        "quality": ["flake8", "mypy", "types-tensorflow", "black"],
    },
    include_package_data=True,
    package_dir={"DeepGlow": "DeepGlow"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
)
