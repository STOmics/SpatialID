#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 10:48
# @Author  : zhangchao
# @File    : transfer.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
from typing import Union

import torch
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt

import torch_geometric
from anndata import AnnData

from spatialid import reader, DnnTrainer
from spatialid.trainer import Base, SpatialTrainer


class Transfer(Base):
    """Implementation of Spatial-ID

    :param spatial_data: Spatial transcriptome data, which be saved in `h5ad` format.
    :param single_data: Single cell transcriptome data, which be saved in `h5ad` format.
    :param output_path: The annotated data and model save path.
    :param device: If the GPU is available, and device > -1 the GPU will be used. If device=-1, only 'CPU' will be used.
    """

    def __init__(self,
                 spatial_data: Union[str, AnnData],
                 single_data: Union[str, AnnData] = None,
                 output_path: str = "./output",
                 device=1):
        super(Transfer, self).__init__(device=device)
        self.output_path = output_path
        self.device_type = device
        self.save_sc = osp.join(self.output_path, "learn_sc_dnn.bgi")
        os.makedirs(os.path.dirname(self.save_sc), exist_ok=True)
        self.save_st = osp.join(self.output_path, "annotation.bgi")
        os.makedirs(os.path.dirname(self.save_st), exist_ok=True)

        self.sc_data = None if single_data is None else reader(single_data)
        self.st_data = reader(spatial_data)

    @staticmethod
    def filter(adata,
               filter_mt=True,
               min_cell=300,
               min_gene=10,
               max_cell=98.):
        if filter_mt:
            adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-", "Mt-"))
            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
            adata = adata[adata.obs["pct_counts_mt"] < 10].copy()
        if min_cell > 0:
            sc.pp.filter_cells(adata, min_counts=min_cell)
        if min_gene > 0:
            sc.pp.filter_genes(adata, min_cells=min_gene)
        if max_cell < 100:
            assert "total_counts" in adata.obs_keys()
            max_count = np.percentile(adata.obs["total_counts"], max_cell)
            sc.pp.filter_cells(adata, max_counts=max_count)

    def learn_sc(self,
                 sc_data=None,
                 filter_mt=True,
                 min_cell=300,
                 min_gene=10,
                 max_cell=98.,
                 ann_key="celltype",
                 marker_genes=None,
                 batch_size=4096,
                 epoch=200,
                 lr=3e-4,
                 weight_decay=1e-6,
                 gamma=2,
                 alpha=.25,
                 reduction="mean"):
        """Learning single cell type using `DNN` model.

        :param sc_data: Single cell transcriptome data, which be saved in `h5ad` format.
        :param filter_mt: Whether to filter MT- genes. default, True
        :param min_cell: Whether to filter genes. default, 10
        :param min_gene: Whether to filter cells. default, 300
        :param max_cell: Whether to filter cells, Range: (0, 100). default, 98.0
        :param ann_key: The annotation key, which should be saved in `*.obs_keys()`
        :param marker_genes: Whether to use marker list data to train the model. If None, all data is used to train the model. Default, None.
        :param batch_size:
        :param epoch:
        :param lr: learning rate
        :param weight_decay:
        :param gamma:
        :param alpha:
        :param reduction:
        :return:
        """
        if sc_data is None:
            sc_data = self.sc_data
        sc_data.var_names_make_unique()
        assert sc_data is not None, ValueError("Error, `sc_data` can not be `None`!")
        assert batch_size <= sc_data.shape[0], "Error, Batch size cannot be larger than the data set row."
        self.filter(sc_data, filter_mt, min_cell, min_gene, max_cell)

        input_dims = sc_data.shape[1] if marker_genes is None else len(marker_genes)

        trainer = DnnTrainer(input_dims=input_dims,
                             label_names=sc_data.obs[ann_key].cat.categories,
                             device=self.device_type,
                             lr=lr,
                             weight_decay=weight_decay,
                             gamma=gamma,
                             alpha=alpha,
                             reduction=reduction,
                             save_path=self.save_sc)
        genes = sc_data.var_names.tolist() if marker_genes is None else marker_genes
        trainer.train(
            data=sc_data,
            ann_key=ann_key,
            marker_genes=genes,
            batch_size=batch_size,
            epochs=epoch
        )

    @torch.no_grad()
    def sc2st(self,
              sc_pht=None):
        """Transfer single cell type to spatial data

        :param sc_pht: Pre-trained `DNN` model
        :return:
        """
        if sc_pht is None:
            sc_pht = self.save_sc
        ckpt = self.load_checkpoint(sc_pht)

        marker_genes = ckpt["marker_genes"]
        gene_indices = self.st_data.var_names.get_indexer(marker_genes)
        if sp.issparse(self.st_data.X):
            data_x = self.st_data.X.toarray()
        else:
            data_x = self.st_data.X
        data_x = np.pad(data_x, ((0, 0), (0, 1)))[:, gene_indices]
        norm_factor = np.linalg.norm(data_x, axis=1, keepdims=True)
        norm_factor[norm_factor == 0] = 1
        dnn_inputs = torch.Tensor(data_x / norm_factor).split(ckpt["batch_size"])

        sc_model = ckpt["model"]
        sc_model.to(self.device)
        sc_model.eval()

        # Inference with DNN model.
        dnn_predictions = []
        for batch_idx, inputs in enumerate(dnn_inputs):
            inputs = inputs.to(self.device)
            outputs = sc_model(inputs)
            dnn_predictions.append(outputs.detach().cpu().numpy())
        label_names = ckpt['label_names']
        self.st_data.obsm['pseudo_label'] = np.concatenate(dnn_predictions)
        self.st_data.obs['pseudo_class'] = pd.Categorical(
            [label_names[i] for i in self.st_data.obsm['pseudo_label'].argmax(1)])
        self.st_data.uns['pseudo_classes'] = label_names

    def annotation(self,
                   pca_dim=200,
                   n_neigh=30,
                   edge_weight=True,
                   epochs=200,
                   lr=0.01,
                   weight_decay=0.0001,
                   w_cls=20.,
                   w_dae=1.,
                   w_gae=1.,
                   show_results=True):
        """Fine tune the final annotation results, using Graph model.

        :param pca_dim: PCA dims, default=200
        :param n_neigh: neighbors number, default=30
        :param edge_weight: Add edge weight to the graph model, default=True
        :param epochs: GCN training epochs, default=200
        :param lr: learning rate
        :param weight_decay:
        :param w_cls: class num weight, default=20
        :param w_dae: dnn weight
        :param w_gae: gcn weight
        :param show_results: Whether to show the annotation results
        :return:
        """

        self.filter(self.st_data)
        print('  After Preprocessing Data Info: %d cells × %d genes.' % (self.st_data.shape[0], self.st_data.shape[1]))

        if sp.issparse(self.st_data.X):
            self.st_data.X = self.st_data.X.toarray()

        sc.pp.normalize_per_cell(self.st_data, counts_per_cell_after=1e4)
        sc.pp.log1p(self.st_data)
        self.st_data.X = (self.st_data.X - self.st_data.X.mean(0)) / (self.st_data.X.std(0) + 1e-20)
        gene_mat = torch.Tensor(self.st_data.X)
        u, s, v = torch.pca_lowrank(gene_mat, pca_dim)
        gene_mat = torch.matmul(gene_mat, v)
        assert 'spatial' in self.st_data.obsm_keys(), "Error, can not found `spatial` in `st_data.obsm_keys()` list."
        cell_coo = torch.Tensor(self.st_data.obsm['spatial'])

        data = torch_geometric.data.Data(x=gene_mat, pos=cell_coo)
        data = torch_geometric.transforms.KNNGraph(k=n_neigh, loop=True)(data)

        assert 'pseudo_label' in self.st_data.obsm_keys(), "Error, can not found `pseudo_label` in `st_data.obsm_keys()` list, please run `sc2st()` first!"
        data.y = torch.Tensor(self.st_data.obsm['pseudo_label'])

        # Make distances as edge weights.
        if edge_weight:
            data = torch_geometric.transforms.Distance()(data)
            data.edge_weight = 1 - data.edge_attr[:, 0]
        else:
            data.edge_weight = torch.ones(data.edge_index.size(1))

        # Train self-supervision model.
        input_dim = data.num_features
        assert 'pseudo_classes' in self.st_data.uns_keys(), "Error, can not found `pseudo_classes` in `st_data.uns_keys()` list, please run `sc2st()` first!"
        num_classes = len(self.st_data.uns['pseudo_classes'])

        trainer = SpatialTrainer(input_dim, num_classes, lr=lr, weight_decay=weight_decay, device=self.device_type)
        trainer.train(data, epochs, w_cls, w_dae, w_gae)
        trainer.save_checkpoint(self.save_st)

        # Inference.
        predictions = trainer.infer(data)
        celltype_pred = pd.Categorical([self.st_data.uns['pseudo_classes'][i] for i in predictions])

        # Save results.
        result = pd.DataFrame({'cell': self.st_data.obs_names.tolist(), 'celltype_pred': celltype_pred})
        result.to_csv(osp.join(self.output_path, "model.csv"), index=False)
        self.st_data.obs['celltype_pred'] = pd.Categorical(celltype_pred)

        del self.st_data.uns
        self.st_data.write_h5ad(osp.join(self.output_path, "annotation.h5ad"))

        # Save visualization.
        if show_results:
            spot_size = 30
            pseudo_top100 = self.st_data.obs['pseudo_class'].to_numpy()
            other_classes = list(pd.value_counts(self.st_data.obs['pseudo_class'])[100:].index)
            pseudo_top100[self.st_data.obs['pseudo_class'].isin(other_classes)] = 'Others'
            self.st_data.obs['pseudo_top100'] = pd.Categorical(pseudo_top100)
            sc.pl.spatial(self.st_data, img_key=None, color=['pseudo_top100'], spot_size=spot_size, show=False)
            plt.savefig(osp.join(self.output_path, "pseudo_top100.pdf"), bbox_inches='tight', dpi=150)
            sc.pl.spatial(self.st_data, img_key=None, color=['celltype_pred'], spot_size=spot_size, show=False)
            plt.savefig(osp.join(self.output_path, "celltype_pred.pdf"), bbox_inches='tight', dpi=150)
        print("\nSpatialID successfully finished the power source annotation!")
