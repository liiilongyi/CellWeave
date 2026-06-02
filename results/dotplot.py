import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

from utils import clustering

np.random.seed(2026)

# =========================
# 0) 读数据
# =========================
#adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 用来聚类（embedding）
#adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")  # 用来画热图（表达）

adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 有 emb_latent_omics1 分群结果（或 embedding）
adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")  # 有原始稀疏表达

#adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 有 emb_latent_omics1 分群结果（或 embedding）
#adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")  # 有原始稀疏表达

# gene 名设为 var_names
adata_omics1.var["gene"] = adata_omics1.var["gene"].astype(str)
adata_omics1.var_names = adata_omics1.var["gene"]
adata_omics1.var_names_make_unique()

# =========================
# 1) 在 adata 上聚类（18类），再把标签拷贝到 adata_omics1
# =========================
tool = "mclust"
clustering(adata, key="emb_latent_omics1", add_key="emb_latent_omics1", n_clusters=7, method=tool, use_pca=True)
adata.obs["emb_latent_omics1"] = adata.obs["emb_latent_omics1"].astype("category")

group_key = "emb_latent_omics1"

# 用 reindex：找不到的会变 NaN，不会抛异常
aligned = adata.obs[group_key].reindex(adata_omics1.obs_names)

# 看看有多少对不上
n_miss = aligned.isna().sum()
print("obs_names missing in adata:", n_miss, "/", adata_omics1.n_obs)

# 写回去
adata_omics1.obs[group_key] = aligned.values
adata_omics1.obs[group_key] = adata_omics1.obs[group_key].astype("category")


# 1) 规范化（dotplot/marker 都建议用 log1p-normalized）
adata_de = adata_omics1.copy()
sc.pp.normalize_total(adata_de, target_sum=1e4)
sc.pp.log1p(adata_de)

# 2) 找 marker（每个 cluster top N）
sc.tl.rank_genes_groups(
    adata_de,
    groupby=group_key,
    method="wilcoxon",
    use_raw=False,
    n_genes=200
)

# 取每个 cluster 的 top_n marker，合并去重
top_n = 2  # 你想更丰富可以改 8~15
df_rank = sc.get.rank_genes_groups_df(adata_de, group=None)  # all groups
# 每个 group 取 top_n
markers = (
    df_rank.sort_values(["group", "scores"], ascending=[True, False])
           .groupby("group")
           .head(top_n)["names"]
           .dropna()
           .astype(str)
           .unique()
           .tolist()
)

print(f"Selected {len(markers)} markers for dotplot.")

# 3) dotplot（更好看一点的参数）
sc.pl.dotplot(
    adata_de,
    var_names=markers,
    groupby=group_key,
    standard_scale="var",       # 每个基因按列标准化，颜色更可比
    dot_max=0.6,                # 限制点大小上限，避免太挤
    dot_min=0.05,
    swap_axes=True,             # 基因在 y 轴（更常见、更清晰）
    dendrogram=True,            # cluster 做层次聚类排序
    figsize=(10, 6),
    show=True
)