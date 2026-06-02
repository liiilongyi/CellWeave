import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData
import matplotlib.pyplot as plt
from utils import filter_bins_keep_genes

#adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
#adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
#adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
#adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")


#adata2 = sc.read_h5ad("<INPUT_OR_OUTPUT_H5AD>")
adata2 = sc.read_h5ad("<BIN_EXPRESSION_H5AD>")
#adata2 = sc.read_h5ad("<BIN_EXPRESSION_H5AD>")
#adata2 = sc.read_h5ad("<BIN_EXPRESSION_H5AD>")
#adata2 = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")

adata2 = filter_bins_keep_genes(adata2)
#print(adata2)
#common_bins = adata.obs_names.intersection(adata2.obs_names)

#adata = adata[common_bins].copy()
#adata2 = adata2[common_bins].copy()

#assert np.all(adata.obs_names == adata2.obs_names)

#output = "<CELL_LEVEL_EXPRESSION_H5AD>"
output = "<CELL_LEVEL_EXPRESSION_H5AD>"
#output = "<INPUT_OR_OUTPUT_H5AD>"
# ===== 配置 =====
label_col = "CellWeave"  # 你的 segmentation 列名

# 1) 取出 segmentation label，忽略 0（背景）
labels_raw = adata.obs[label_col]
# 如果是字符串 / 混合类型，先转成数值，非法的变为 NaN
labels_num = pd.to_numeric(labels_raw, errors="coerce")
# mask: 只保留在 cell 内的 bin（label != 0 且 非 NaN）
mask = (labels_num != 0) & (~np.isnan(labels_num))
mask = mask.to_numpy()

print("总 bin 数:", adata.n_obs)
print("在 cell 内的 bin 数:", mask.sum())

# 这些是每个 bin 对应的 cell ID（非 0）
labels_nonzero = labels_num[mask].astype(np.int64).to_numpy()

# 2) 把 cell ID 压缩成 0..n_cells-1 的编码，方便矩阵运算
codes, uniques = pd.factorize(labels_nonzero)  # codes: 0..n_cells-1, uniques: 原始 cell ID
n_cells = len(uniques)
print("细胞数 (非 0 segment):", n_cells)

# 3) 表达矩阵：对每个 cell 汇总（按行求和）
X = adata2.X
if not sparse.isspmatrix_csr(X):
    X = sparse.csr_matrix(X)

# 只保留属于 cell 的 bin
X_sub = X[mask]                      # (n_bins_in_cells, n_genes)
n_bins_in_cells = X_sub.shape[0]

# 构造一张 (n_bins_in_cells x n_cells) 的 one-hot 稀疏矩阵
row_idx = np.arange(n_bins_in_cells, dtype=np.int64)
col_idx = codes                      # 每个 bin 对应哪个 cell
data = np.ones(n_bins_in_cells, dtype=np.float32)

one_hot = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_bins_in_cells, n_cells))

# cell 表达 = one_hot^T @ X_sub  ->  (n_cells x n_genes)
X_cells = one_hot.T @ X_sub          # 依然是 sparse matrix

print("聚合后的表达矩阵形状:", X_cells.shape)

# 4) 计算每个 cell 的 px / py 质心
px_bins = adata.obs["px"].to_numpy()[mask]
py_bins = adata.obs["py"].to_numpy()[mask]

# 每个 cell 有多少个 bin
cell_counts = np.bincount(codes)     # 长度 = n_cells

px_cells = np.bincount(codes, weights=px_bins) / cell_counts
py_cells = np.bincount(codes, weights=py_bins) / cell_counts

# 5) 构建新的 obs / var / AnnData
# obs：index 用原始 segmentation ID，方便追溯
obs_new = pd.DataFrame({
    "seed": uniques,  # 原始 segmentation 的数字 ID
    "px": px_cells,
    "py": py_cells,
})
obs_new.index = obs_new["seed"].astype(str)  # 作为 obs index
obs_new = obs_new.drop(columns=["seed"])

# var：直接沿用原始的 gene 信息
var_new = adata2.var.copy()

adata_cells = AnnData(X=X_cells, obs=obs_new, var=var_new)

print(adata_cells)
print(adata_cells.X.shape)
print(adata_cells.obs.head())

Xc = adata_cells.X

# 确保是 CSR，这样 indptr 表示的是按行的非零数
if not sparse.isspmatrix_csr(Xc):
    Xc = Xc.tocsr()

nonzero_per_cell = np.diff(Xc.indptr)

total_cells = adata_cells.n_obs
print("Total cells:", total_cells)
print("Cells with 0 expressed genes:", (nonzero_per_cell == 0).sum())
print(f"Fraction: {(nonzero_per_cell == 0).sum() / total_cells:.4f}")
print("Mean non-zero genes per cell:", nonzero_per_cell.mean())
print("Median:", np.median(nonzero_per_cell))
print("Min:", nonzero_per_cell.min())
print("Max:", nonzero_per_cell.max())

nonzero = nonzero_per_cell

# 统计频数
values, counts = np.unique(nonzero, return_counts=True)

print("=== Frequency Table (first 30 items) ===")
for v, c in zip(values[:30], counts[:30]):
    print(f"{v}: {c}")

print("Total unique values:", len(values))
plt.figure(figsize=(8, 5))
plt.hist(nonzero, bins=50, color='steelblue', alpha=0.8)
plt.xlabel("Number of expressed genes per cell")
plt.ylabel("Cell count")
plt.title("Distribution of non-zero genes per cell")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.hist(nonzero, bins=100, color='orange', alpha=0.75)
plt.yscale("log")
plt.xlabel("Number of expressed genes per cell")
plt.ylabel("Cell count (log scale)")
plt.title("Distribution of expressed genes per cell (log-scaled)")
plt.tight_layout()
plt.show()



adata_cells.write_h5ad(output)
#adata_cells.write_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")
