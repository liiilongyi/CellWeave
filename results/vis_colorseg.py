import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from utils import clustering

adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 用来聚类（embedding）
adata2 = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
seed_key_in_bins = "<BIN_SEGMENTATION_COLUMN>"

tool = "mclust"
clustering(adata, key="emb_latent_omics1", add_key="emb_latent_omics1", n_clusters=24, method=tool, use_pca=True)   # brain 24 Ovarian 7
adata.obs["emb_latent_omics1"] = adata.obs["emb_latent_omics1"].astype("category")

# =========================
# 2) 建立 seed -> cluster 的映射
#    adata.obs_names 就是 seed（你说 seed 是细胞编号）
# =========================
# seed 可能是字符串，强制转 int
seed_ids = pd.Index(adata.obs_names).astype(str)

# 转成 int（如果 seed 里有非数字会报错，方便你及时发现问题）
seed_ids_int = seed_ids.astype(int).to_numpy()

# cluster 可能是 '0','1'... 或 int 类别，这里统一转 int
cluster_cat = adata.obs["emb_latent_omics1"].astype(str).astype(int).to_numpy()

seed_to_cluster = dict(zip(seed_ids_int, cluster_cat))

n_clusters = adata.obs["emb_latent_omics1"].cat.categories.size
print(f"[INFO] n_clusters = {n_clusters}")
print(f"[INFO] adata cells = {adata.n_obs}")

# =========================
# 3) 取 adata2 中每个 bin 的 seed，并映射成 cluster
#    seed==0 认为是背景
# =========================
bin_seed = adata2.obs[seed_key_in_bins].to_numpy()
bin_seed = bin_seed.astype(np.int64)

# 先全置为 -1（背景/未匹配）
bin_cluster = np.full(bin_seed.shape[0], -1, dtype=np.int16)

# 只处理 seed!=0 的 bin
mask_cell = bin_seed != 0
seed_nonzero = bin_seed[mask_cell]

# 把 seed 映射成 cluster（若 seed 在 adata 中不存在，就保持 -1）
# 用 pandas map 比 python 循环快很多
mapped = pd.Series(seed_nonzero).map(seed_to_cluster).to_numpy()

# mapped 可能有 NaN（表示 seed 不在 adata 中）
valid = ~pd.isna(mapped)
bin_cluster_idx = np.where(mask_cell)[0]

bin_cluster[bin_cluster_idx[valid]] = mapped[valid].astype(np.int16)

print(f"[INFO] bins total = {adata2.n_obs}")
print(f"[INFO] bins with seed!=0 = {mask_cell.sum()}")
print(f"[INFO] bins assigned cluster = {(bin_cluster >= 0).sum()}")
print(f"[INFO] bins unmatched (seed!=0 but not in adata) = {mask_cell.sum() - (bin_cluster >= 0).sum()}")

# =========================
# 4) 解析 barcode 得到 (row, col)，构建 RGB 图
#    barcode 格式：s_002um_00019_01375-1
# =========================
idx_series = pd.Index(adata2.obs_names).astype(str).to_series(index=adata2.obs_names)
rc = idx_series.str.extract(r"s_002um_(\d+)_(\d+)-1")

if rc.isnull().any().any():
    bad = rc[rc.isnull().any(axis=1)].index[:5].tolist()
    raise ValueError(f"[ERROR] barcode 解析失败（示例前5个）：{bad}，请检查 obs_names 格式！")

rows = rc[0].astype(np.int32).to_numpy()
cols = rc[1].astype(np.int32).to_numpy()

# 平移到 0-based
row_min, col_min = rows.min(), cols.min()
rows0 = rows - row_min
cols0 = cols - col_min

H = int(rows0.max() + 1)
W = int(cols0.max() + 1)
print(f"[INFO] canvas size: H={H}, W={W}")

# =========================
# 5) 颜色映射：每个 cluster 一个颜色，背景黑色
# =========================
# 24 类建议用 'turbo' 或 'gist_ncar' 更好区分
cmap = plt.cm.get_cmap("turbo", n_clusters)

# 生成 [n_clusters,3] 的 uint8 颜色表
colors = (cmap(np.arange(n_clusters))[:, :3] * 255).astype(np.uint8)

# 初始化 RGB 图（黑色背景）
mask_img = np.zeros((H, W, 3), dtype=np.uint8)

# 只给 cluster>=0 的 bin 上色
ok = bin_cluster >= 0
r = rows0[ok]
c = cols0[ok]
cl = bin_cluster[ok].astype(np.int32)

mask_img[r, c] = colors[cl]

# =========================
# 6) 显示 / 保存
# =========================
plt.figure(figsize=(12, 10))
plt.imshow(mask_img, origin="upper")
plt.axis("off")
plt.tight_layout()

#out_png = "<OUTPUT_FIGURE_FILE>"
out_png = "<OUTPUT_FIGURE_FILE>"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()



print("[DONE] saved:", out_png)

# （可选）如果你还想把 cluster 标签写回 adata2（很大，保存会变慢/变大）
# adata2.obs["cell_cluster24"] = bin_cluster
# adata2.write_h5ad("<SEGMENTED_BIN_H5AD>")
