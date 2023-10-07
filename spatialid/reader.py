#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 11:02
# @Author  : zhangchao
# @File    : reader.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import os
import scanpy as sc
from anndata import AnnData


def reader(data):
    if isinstance(data, AnnData):
        return data
    elif isinstance(data, str) and data.endswith("h5ad"):
        assert os.path.exists(data), ValueError(f"There was no data path: `{data}`!")
        return sc.read_h5ad(data)
    else:
        raise ValueError(f"Got an invalid data format, only support `str` and `AnnData`!")
