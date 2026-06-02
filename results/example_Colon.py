import os
import numpy as np
import torch
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
os.environ['R_HOME'] = "<R_HOME_PATH>"

embedding = np.load("<EMBEDDING_NPY>")
outpdf = "<OUTPUT_FIGURE_FILE>"

adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")
coords = np.vstack([
    adata_omics1.obs["px"].values,
    adata_omics1.obs["py"].values
]).T
adata_omics1.obsm["spatial"] = coords

adata_omics1.var_names_make_unique()

datatype = '10x'
random_seed = 2026
fix_seed(random_seed)


sc.pp.filter_genes(adata_omics1, min_cells=10) #
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=200)
#adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=200)
#adata_omics1.obsm['feat'] = np.zeros((3484, 30), dtype=np.float32)



data = construct_neighbor_graph(adata_omics1, adata_omics1, datatype=datatype)
print("kneighbors_graph...")
feature_graph_emb = kneighbors_graph(embedding, 20, mode="connectivity", metric="correlation", include_self=False)
data['adj_emb'] = feature_graph_emb

model = Train_spaLLM(data, datatype=datatype, device=device, embedding=embedding)
print("train...")
output = model.train(epochs=500)

adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()

adata.obsm['alpha_omics1'] = output['alpha_omics1']
adata.obsm['alpha_att1']= output['alpha_att1']
adata.obsm['alpha_att2']= output['alpha_att2']

print(adata.obsm['emb_latent_omics1'])
print(adata)
adata.write("<CELL_LEVEL_EMBEDDING_H5AD>")

tool = 'mclust'
clustering(adata, key='emb_latent_omics1', add_key='emb_latent_omics1', n_clusters=13, method=tool, use_pca=True)
adata_plt = adata.copy()
adata_plt.obs['emb_latent_omics1'] = adata.obs['emb_latent_omics1'].astype('category')



print(adata.obsm['emb_latent_omics1'])
print(adata.obsm['emb_latent_omics1'].shape)


# visualization


fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
sc.pp.neighbors(adata_plt, use_rep='emb_latent_omics1', n_neighbors=10)
sc.tl.umap(adata_plt)

sc.pl.umap(adata_plt, color='emb_latent_omics1', ax=ax_list[0], title='cluster_13', s=30, show=False)
sc.pl.embedding(adata_plt, basis='spatial', color='emb_latent_omics1', ax=ax_list[1], title='cluster_13', s=25, show=False)

plt.tight_layout(w_pad=0.3)
plt.savefig(outpdf, dpi=600, bbox_inches="tight")
plt.show()








