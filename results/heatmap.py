import scanpy as sc
import scipy.sparse as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import clustering

# 1) 读入两个对象
adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 有 emb_latent_omics1 分群结果（或 embedding）
adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")  # 有原始稀疏表达

#adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 有 emb_latent_omics1 分群结果（或 embedding）
#adata_omics1 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")  # 有原始稀疏表达

adata_omics1.var["gene"] = adata_omics1.var["gene"].astype(str)
adata_omics1.var_names = adata_omics1.var["gene"]
adata_omics1.var_names_make_unique()

tool = 'mclust'
clustering(adata, key='emb_latent_omics1', add_key='emb_latent_omics1', n_clusters=7, method=tool, use_pca=True)
adata.obs['emb_latent_omics1'] = adata.obs['emb_latent_omics1'].astype('category')

group_key = "emb_latent_omics1"

print("adata:", adata.shape)
print("adata_omics1:", adata_omics1.shape)

# 2) 确保 obs 一一对应：推荐用 obs_names 对齐（最稳）
# 如果你确认行顺序完全一致，也可以直接 values 赋值
if adata.obs_names.equals(adata_omics1.obs_names):
    adata_omics1.obs[group_key] = adata.obs[group_key].values
else:
    # 用 barcode/obs_names 对齐拷贝
    adata_omics1.obs[group_key] = adata.obs.loc[adata_omics1.obs_names, group_key].values

adata_omics1.obs[group_key] = adata_omics1.obs[group_key].astype("category")
print("clusters:", adata_omics1.obs[group_key].cat.categories.tolist())
print("cluster counts:\n", adata_omics1.obs[group_key].value_counts())

# 3) 准备 counts layer + log1p（在 adata_omics1 上做）
# 如果 adata_omics1.X 就是 counts（你展示的基本是 0/1 计数），那非常适合
if "counts" not in adata_omics1.layers:
    adata_omics1.layers["counts"] = adata_omics1.X.copy()

# 用 counts 构建 log1p 表达供 DE/热图使用
adata_de = adata_omics1.copy()
adata_de.X = adata_de.layers["counts"].copy()

# normalize + log1p（标准做法）
sc.pp.normalize_total(adata_de, target_sum=1e4)
sc.pp.log1p(adata_de)

# （可选但推荐）筛 HVG，避免 38606 基因太多导致 DE 慢/噪声大
# 你也可以把 n_top_genes 调大/调小
#sc.pp.highly_variable_genes(adata_de, n_top_genes=3000, flavor="seurat_v3")
#adata_de = adata_de[:, adata_de.var["highly_variable"]].copy()
#print("After HVG:", adata_de.shape)

# 4) 差异分析（每个 cluster vs rest）
sc.tl.rank_genes_groups(
    adata_de,
    groupby=group_key,
    method="wilcoxon",
    n_genes=200,
    use_raw=False,
    key_added="rank_" + group_key
)

# 5) 取每个 cluster top marker genes
n_top = 15
df = sc.get.rank_genes_groups_df(adata_de, group=None, key="rank_" + group_key)

# 过滤：显著 + 正向
df = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0)].copy()

# 如果过滤后为空，说明统计太严/数据稀疏；就放宽（至少保证能画）
if df.shape[0] == 0:
    df = sc.get.rank_genes_groups_df(adata_de, group=None, key="rank_" + group_key)

top_df = (
    df.sort_values(["group", "scores"], ascending=[True, False])
      .groupby("group", observed=False)
      .head(n_top)
)

gene_list = pd.unique(top_df["names"]).tolist()
print(f"Heatmap genes: {len(gene_list)}")
print(gene_list[:30])

print(top_df["group"].value_counts())
genes = gene_list
X = adata_de[:, genes].X
if sp.issparse(X):
    X = X.toarray()

df = pd.DataFrame(X, columns=genes, index=adata_de.obs_names)
df["cluster"] = adata_de.obs[group_key].astype(str).values

mean_expr = df.groupby("cluster").mean()
print(mean_expr.shape)          # (n_clusters, n_genes)
print(mean_expr.loc["2"].head())
exit()

# 6) 画热图（cluster 平均表达）
sc.pl.matrixplot(
    adata_de,
    var_names=gene_list,
    groupby=group_key,
    use_raw=False,
    standard_scale="var",
    cmap="viridis",
    dendrogram=False,
    figsize=(max(10, len(gene_list) * 0.25), 5)
)
plt.show()
exit()


# 7) 导出 marker 表
top_df.to_csv("<OUTPUT_CSV_FILE>", index=False)
print("<OUTPUT_CSV_FILE>")
