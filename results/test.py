import os
import numpy as np
import torch
import pandas as pd
import scanpy as sc

from preprocess import fix_seed
from preprocess import clr_normalize_each_cell, pca
from preprocess import construct_neighbor_graph
from spaLLM_util import Train_spaLLM
from utils import clustering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import kneighbors_graph

device = torch.device('cuda')

adata = sc.read_h5ad("<INPUT_OR_OUTPUT_H5AD>")
print(adata)

tool = 'mclust'
clustering(adata, key='emb_latent_omics1', add_key='emb_latent_omics1', n_clusters=18, method=tool, use_pca=True)
adata_plt = adata.copy()
adata_plt.obs['emb_latent_omics1'] = adata.obs['emb_latent_omics1'].astype('category')



print(adata.obsm['emb_latent_omics1'])
print(adata.obsm['emb_latent_omics1'].shape)


# visualization


fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
sc.pp.neighbors(adata_plt, use_rep='emb_latent_omics1', n_neighbors=10)
sc.tl.umap(adata_plt)

sc.pl.umap(adata_plt, color='emb_latent_omics1', ax=ax_list[0], title='cluster_15', s=30, show=False)
sc.pl.embedding(adata_plt, basis='spatial', color='emb_latent_omics1', ax=ax_list[1], title='cluster_15', s=25, show=False)

plt.tight_layout(w_pad=0.3)
plt.show()

# 1. 处理 cluster 列
adata.obs['cluster'] = adata.obs['emb_latent_omics1'].astype(str)
adata.obs['cluster'] = adata.obs['cluster'].astype('category')

# 2. 做 DE
sc.tl.rank_genes_groups(
    adata,
    groupby='cluster',
    method='wilcoxon',
    n_genes=50,
    key_added='rank_cluster'
)

# 3. 画 rank_genes_groups 自带的热图
sc.pl.rank_genes_groups_heatmap(
    adata,
    key='rank_cluster',
    groupby='cluster',
    n_genes=10,
    standard_scale='var',
    show_gene_labels=True
)
