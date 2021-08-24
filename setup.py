# Copyright (c) 2021 Sartorius AG
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup

import dime

with open("README.md", "r") as fp:
    long_description = fp.read()

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

setup(
    name="dime-pytorch",
    version=dime.__version__,
    author="Rickard Sjoegren",
    author_email="rickard.sjoegren@sartorius.com",
    url="https://github.com/sartorius-research/dime.pytorch",
    py_modules=["dime"],
    description="PyTorch-implementation of the DIME method to detect out-of-distribution "
                "observations in deep learning.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="MIT",
    classifiers=classifiers,
    python_requires=">=3.6",
    install_requires=[
        'numpy',
        'torch'
    ]
)
