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

#adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 有 emb_latent_omics1 分群结果（或 embedding）
#adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")  # 有原始稀疏表达

adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 有 emb_latent_omics1 分群结果（或 embedding）# part2
#adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")  # 有原始稀疏表达
adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")  # 有原始稀疏表达

# gene 名设为 var_names
adata_omics1.var["gene"] = adata_omics1.var["gene"].astype(str)
adata_omics1.var_names = adata_omics1.var["gene"]
adata_omics1.var_names_make_unique()

# =========================
# 1) 在 adata 上聚类（18类），再把标签拷贝到 adata_omics1
# =========================
tool = "mclust"
clustering(adata, key="emb_latent_omics1", add_key="emb_latent_omics1", n_clusters=13, method=tool, use_pca=True)
adata.obs["emb_latent_omics1"] = adata.obs["emb_latent_omics1"].astype("category")

group_key = "emb_latent_omics1"

# 对齐 obs_names（最稳）
if adata.obs_names.equals(adata_omics1.obs_names):
    adata_omics1.obs[group_key] = adata.obs[group_key].values
else:
    adata_omics1.obs[group_key] = adata.obs.loc[adata_omics1.obs_names, group_key].values

adata_omics1.obs[group_key] = adata_omics1.obs[group_key].astype("category")

print("clusters:", adata_omics1.obs[group_key].cat.categories.tolist())
print("cluster counts:\n", adata_omics1.obs[group_key].value_counts())

# =========================
# 2) normalize + log1p（仿照你参考代码）
# =========================
adata_de = adata_omics1.copy()

# 如果你确定 X 就是 raw counts，这样做最合理
sc.pp.normalize_total(adata_de, target_sum=1e4)
sc.pp.log1p(adata_de)

# =========================
# 3) rank_genes_groups（仿照你参考代码）
# =========================
sc.tl.rank_genes_groups(adata_de, groupby=group_key, method="wilcoxon", use_raw=False, n_genes=200)

ranked_genes = adata_de.uns["rank_genes_groups"]
clusters = ranked_genes["names"].dtype.names  # e.g. ('0','1',...,'17')

top_n = 10   # 你参考代码是5；如果觉得基因太少不好看，建议改成 8~15
top_genes = []
for clust in clusters:
    genes = ranked_genes["names"][clust][:top_n]
    top_genes.extend(list(genes))

# ✅ 保序去重（不要用 set，会打乱顺序）
top_genes = list(pd.unique(top_genes))
print(f"Selected {len(top_genes)} genes for custom heatmap.")

# =========================
# 4) 构建表达矩阵 DataFrame（只对 top_genes 子矩阵 toarray，避免爆内存）
# =========================
# 只取 top_genes 对应的列
gene_idx = adata_de.var_names.get_indexer(top_genes)
valid = gene_idx >= 0
top_genes = [g for g, ok in zip(top_genes, valid) if ok]
gene_idx = gene_idx[valid]

X_sub = adata_de.X[:, gene_idx]  # (n_cells, n_top_genes)
if sp.issparse(X_sub):
    X_sub = X_sub.toarray()

expr_df = pd.DataFrame(X_sub, columns=top_genes, index=adata_de.obs_names)
expr_df["cluster"] = adata_de.obs[group_key].values

# =========================
# 5) 每个 cluster 求平均表达 + Z-score（仿照你参考代码）
# =========================
mean_expr = expr_df.groupby("cluster", observed=False).mean()

# 每个基因做 z-score（列标准化）
mean_expr_z = (mean_expr - mean_expr.mean(axis=0)) / (mean_expr.std(axis=0) + 1e-8)
mean_expr_z = mean_expr_z.fillna(0)

# =========================
# 6) 对角线友好的基因排序（核心修改）
# =========================

# 1) 对每个基因，找它 Z-score 最大的 cluster
dominant_cluster = mean_expr_z.idxmax(axis=0)
# index: gene, value: cluster

# 2) 把 cluster 转成有序的 rank（0,1,2,...）
cluster_order = {c: i for i, c in enumerate(mean_expr_z.index)}
dominant_cluster_rank = dominant_cluster.map(cluster_order)

# 3) 同时取该基因的最大 Z-score（显著性强弱）
max_expr = mean_expr_z.max(axis=0)

# 4) 构建排序表
gene_order_df = pd.DataFrame({
    "gene": mean_expr_z.columns,
    "cluster": dominant_cluster.values,
    "cluster_rank": dominant_cluster_rank.values,
    "max_expr": max_expr.values
})

# 5) 排序规则（关键）：
#    - 先按 cluster_rank（决定靠左/靠右）
#    - 再按 max_expr（显著的放前面）
gene_order_df = gene_order_df.sort_values(by=["cluster_rank", "max_expr"], ascending=[True, False])

# 6) 得到新的基因顺序
ordered_genes = gene_order_df["gene"].tolist()

# 7) 按新顺序重排矩阵
mean_expr_z_ordered = mean_expr_z[ordered_genes]

# =========================
# 7) seaborn 热图（蓝-白-红，center=0）+ 保存
# =========================
plt.figure(figsize=(max(10, 0.25 * mean_expr_z_ordered.shape[1]), max(6, 0.35 * mean_expr_z_ordered.shape[0])))

ax = sns.heatmap(
    mean_expr_z_ordered,
    cmap="coolwarm",
    center=0,
    # 建议固定色阶，否则颜色会漂（可按需调）
    vmin=-2, vmax=2,
    xticklabels=True,
    yticklabels=True
)

plt.ylabel("Cluster")
plt.xlabel("Gene")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

#out_pdf = "<SEGMENTATION_VISUALIZATION_FIGURE>"
#out_pdf = "<SEGMENTATION_VISUALIZATION_FIGURE>"
#out_pdf = "<SEGMENTATION_VISUALIZATION_FIGURE>"
out_pdf = "<SEGMENTATION_VISUALIZATION_FIGURE>"
plt.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=600)
plt.show()

print("Saved:", out_pdf)
