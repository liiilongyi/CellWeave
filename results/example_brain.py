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
os.environ['R_HOME'] = "<R_HOME_PATH>"



"""import rpy2.robjects as robjects
robjects.r('''library(mclust)''')"""


adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")
coords = np.vstack([
    adata_omics1.obs["px"].values,
    adata_omics1.obs["py"].values
]).T
adata_omics1.obsm["spatial"] = coords

#adata2 = sc.read_h5ad("<INPUT_OR_OUTPUT_H5AD>")

embedding = np.load("<EMBEDDING_NPY>")

adata_omics1.var_names_make_unique()

datatype = 'Brain'      # need to change
random_seed = 2025
fix_seed(random_seed)


sc.pp.filter_genes(adata_omics1, min_cells=10) #
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=300)
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




exit()




plt.rcParams['figure.figsize'] = (8, 5)
df = pd.DataFrame(columns=['embedding', 'omic', 'label'])
df['embedding'], df['omic'] = adata.obsm['alpha_att1'][:, 0], adata.obsm['alpha_att1'][:, 1]
df['label'] = adata.obs['spaFDG'].values
df = df.set_index('label').stack().reset_index()
df.columns = ['label_spaFDG', 'Modality', 'Weight value']
ax = sns.violinplot(data=df, x='label_spaFDG', y='Weight value', hue='Modality', split=True, inner='quart', linewidth=1, show=False)
ax.set_title('embedding_spa vs omic')
ax.set_xlabel('spaFDG label')
ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')

plt.tight_layout(w_pad=0.05)
save_path = '<LOCAL_DATA_OR_OUTPUT_PATH>'
plt.savefig(save_path + '<OUTPUT_FIGURE_FILE>', format='jpeg', bbox_inches='tight', dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = (8, 5)
df = pd.DataFrame(columns=['embedding', 'omic', 'label'])
df['embedding'], df['omic'] = adata.obsm['alpha_att2'][:, 0], adata.obsm['alpha_att2'][:, 1]
df['label'] = adata.obs['spaFDG'].values
df = df.set_index('label').stack().reset_index()
df.columns = ['label_spaFDG', 'Modality', 'Weight value']
ax = sns.violinplot(data=df, x='label_spaFDG', y='Weight value', hue='Modality', split=True, inner='quart', linewidth=1, show=False)
ax.set_title('embedding_fea vs omic')
ax.set_xlabel('spaFDG label')
ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')

plt.tight_layout(w_pad=0.05)
save_path = '<LOCAL_DATA_OR_OUTPUT_PATH>'
plt.savefig(save_path + '<OUTPUT_FIGURE_FILE>', format='jpeg', bbox_inches='tight', dpi=300)
plt.show()

exit()





plt.rcParams['figure.figsize'] = (5, 3)
df = pd.DataFrame(columns=['RNA', 'protein', 'label'])
df['RNA'], df['protein'] = adata.obsm['alpha'][:, 0], adata.obsm['alpha'][:, 1]
df['label'] = adata.obs['spaLLM'].values
df = df.set_index('label').stack().reset_index()
df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue='Modality', split=True, inner='quart', linewidth=1, show=False)
ax.set_title('RNA vs protein')
ax.set_xlabel('spaLLM label')
ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')

plt.tight_layout(w_pad=0.05)
plt.show()