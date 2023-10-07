#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 9:58
# @Author  : zhangchao
# @File    : spatial.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import remove_self_loops, add_self_loops, negative_sampling


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.gc_feat = GCNConv(in_channels, hidden_channels)
        self.gc_mean = GCNConv(hidden_channels, out_channels)
        self.gc_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.gc_feat(x, edge_index, edge_weight).relu()
        mean = self.gc_mean(x, edge_index, edge_weight)
        logstd = self.gc_logstd(x, edge_index, edge_weight)
        return mean, logstd


def full_block(in_features, out_features, drop_rate=0.2):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=drop_rate)
    )


class SpatialModel(nn.Module):
    def __init__(self, input_dim, num_classes, gae_dim, dae_dim, feat_dim):
        super(SpatialModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gae_dim = gae_dim
        self.dae_dim = dae_dim
        self.feat_dim = feat_dim
        self.fcat_dim = self.dae_dim[1] + self.gae_dim[1]
        self.encoder = nn.Sequential(full_block(self.input_dim, self.dae_dim[0]),
                                     full_block(self.dae_dim[0], self.dae_dim[1]))
        self.decoder = nn.Linear(self.feat_dim, self.input_dim)
        self.vgae = VGAE(GraphEncoder(self.dae_dim[1], self.gae_dim[0], self.gae_dim[1]))
        self.feat_fc_x = nn.Sequential(nn.Linear(self.fcat_dim, self.feat_dim), nn.ELU())
        self.feat_fc_g = nn.Sequential(nn.Linear(self.fcat_dim, self.feat_dim), nn.ELU())
        self.classifier = nn.Linear(self.fcat_dim, self.num_classes)

    def forward(self, x, edge_index, edge_weight):
        feat_x = self.encoder(x)
        feat_g = self.vgae.encode(feat_x, edge_index, edge_weight)
        feat = torch.cat([feat_x, feat_g], 1)
        feat_x = self.feat_fc_x(feat)
        feat_g = self.feat_fc_g(feat)
        x_dec = self.decoder(feat_x)
        dae_loss = F.mse_loss(x_dec, x)
        gae_loss = self.recon_loss(feat_g, edge_weight, edge_index) + 1 / len(x) * self.vgae.kl_loss()
        cls = self.classifier(feat)
        return cls, dae_loss, gae_loss

    def recon_loss(self, z, edge_weight, pos_edge_index, neg_edge_index=None):
        pos_dec = self.vgae.decoder(z, pos_edge_index, sigmoid=False)
        pos_loss = F.binary_cross_entropy_with_logits(pos_dec, edge_weight)
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_dec = self.vgae.decoder(z, neg_edge_index, sigmoid=False)
        neg_loss = -F.logsigmoid(-neg_dec).mean()
        return pos_loss + neg_loss
