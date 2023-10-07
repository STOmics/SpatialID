#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/7 16:09
# @Author  : zhangchao
# @File    : setup.py.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import setuptools
from wheel.bdist_wheel import bdist_wheel

import spatialid

__version__ = spatialid.__version__


class BDistWheel(bdist_wheel):
    def get_tag(self):
        return (self.python_tag, "none", "any")


cmdclass = {
    "bdist_wheel": BDistWheel,
}

requirements = open("requirements.txt").readline()

setuptools.setup(
    name="SpatialID",
    version=__version__,
    author="zhangchao",
    author_email="1623804006@qq.com",
    url="https://github.com/STOmics/SpatialID.git",
    description="Spatial-ID: a cell typing method for spatially resolved transcriptomics via transfer learning and spatial embedding",
    python_requires=">=3.8",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    cmdclass=cmdclass,
)
