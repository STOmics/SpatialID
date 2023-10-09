#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 10:39
# @Author  : zhangchao
# @File    : dnn.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn


class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_rate=0.5):
        super(DNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 2)),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(int(hidden_dim / 2), output_dim),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):
        return self.net(x)
