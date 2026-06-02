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

#adata =  sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")
adata =  sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")

tool = 'mclust'
clustering(adata, key='emb_latent_omics1', add_key='emb_latent_omics1', n_clusters=24, method=tool, use_pca=True)
adata_plt = adata.copy()
adata_plt.obs['emb_latent_omics1'] = adata.obs['emb_latent_omics1'].astype('category')



print(adata.obsm['emb_latent_omics1'])
print(adata.obsm['emb_latent_omics1'].shape)


# visualization


fig, ax_list = plt.subplots(1, 2, figsize=(8.5, 3))
sc.pp.neighbors(adata_plt, use_rep='emb_latent_omics1', n_neighbors=10)
sc.tl.umap(adata_plt)

sc.pl.umap(adata_plt, color='emb_latent_omics1', ax=ax_list[0], title='cluster_15', s=30, show=False)
sc.pl.embedding(adata_plt, basis='spatial', color='emb_latent_omics1', ax=ax_list[1], title='cluster_15', s=25, show=False)

plt.tight_layout(w_pad=0.3)
plt.show()