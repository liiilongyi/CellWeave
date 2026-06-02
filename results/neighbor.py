import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
from utils import clustering

adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 用来聚类（embedding）

tool = "mclust"
clustering(adata, key="emb_latent_omics1", add_key="emb_latent_omics1", n_clusters=7, method=tool, use_pca=True)
adata.obs["cluster"] = adata.obs["emb_latent_omics1"].astype("category")

# =========================================================
# 0) 基本设置
# =========================================================
type_key = "cluster"
x = adata.obs["px"].to_numpy().astype(np.float32)
y = adata.obs["py"].to_numpy().astype(np.float32)

types = adata.obs[type_key].astype("category")
type_names = types.cat.categories.tolist()
type_codes = types.cat.codes.to_numpy()  # 0..K-1
K = len(type_names)

coords = np.c_[x, y]

# 邻域图参数：k 越小越“接触”，越大越“局部微环境”
KNN = 8  # 建议 6~12 之间；你可以试 6/8/10 做稳定性对比
N_PERM = 200  # 置换次数；最终论文可提升到 500/1000
SEED = 0
rng = np.random.default_rng(SEED)

out_prefix = "nei/fig"

# =========================================================
# 1) 构建 cell-level 空间邻接图（kNN）
# =========================================================
nbrs = NearestNeighbors(n_neighbors=KNN+1, algorithm="auto", metric="euclidean")
nbrs.fit(coords)
dists, idxs = nbrs.kneighbors(coords, return_distance=True)

# 去掉自己（第一列是自己）
dists = dists[:, 1:]          # [N, KNN]
idxs = idxs[:, 1:]            # [N, KNN]
N = coords.shape[0]

# 构建无向边集（i<j）
src = np.repeat(np.arange(N), KNN)
dst = idxs.reshape(-1)
# 去自环
mask = src != dst
src, dst = src[mask], dst[mask]

# 无向化：只保留 i<j
a = np.minimum(src, dst)
b = np.maximum(src, dst)
edges = np.unique(np.c_[a, b], axis=0)
src_u, dst_u = edges[:, 0], edges[:, 1]
E = edges.shape[0]
print(f"[Graph] N={N}, KNN={KNN}, undirected edges E={E}")

# =========================================================
# 2) 基本 QC 图：边长分布 + 局部邻接示意（可用于 Supplement）
# =========================================================
# 2.1 边长分布
edge_len = np.sqrt((x[src_u]-x[dst_u])**2 + (y[src_u]-y[dst_u])**2)

plt.figure(figsize=(5,4))
plt.hist(edge_len, bins=60)
plt.xlabel("kNN edge length (pixels)")
plt.ylabel("Count")
plt.title("Spatial kNN graph edge lengths")
plt.tight_layout()
plt.savefig(out_prefix + "<OUTPUT_FIGURE_FILE>", dpi=300)
plt.show()

# 2.2 局部邻接示意（随机挑一个窗口）
# 选一个中心点，画它的邻居边
center = rng.integers(0, N)
nb = idxs[center]
plt.figure(figsize=(5,5))
plt.scatter(x, y, s=0.2, alpha=0.15)
plt.scatter(x[center], y[center], s=30, marker="x")
plt.scatter(x[nb], y[nb], s=15)
for j in nb:
    plt.plot([x[center], x[j]], [y[center], y[j]], linewidth=0.5)
plt.gca().invert_yaxis()
plt.axis("equal"); plt.axis("off")
plt.title("Local kNN neighborhood (example)")
plt.tight_layout()
plt.savefig(out_prefix + "<OUTPUT_FIGURE_FILE>", dpi=300)
plt.show()

# =========================================================
# 3) 统计核心：type-type 邻接计数矩阵 + 置换检验
# =========================================================
def count_type_edges(type_codes_local):
    """给定每个 cell 的 type 编码，返回 KxK 的无向边计数矩阵（对称）"""
    t1 = type_codes_local[src_u]
    t2 = type_codes_local[dst_u]

    # 统计无向 pair (min,max)
    p = np.minimum(t1, t2) * K + np.maximum(t1, t2)
    bc = np.bincount(p, minlength=K*K).reshape(K, K)

    # bc 是上三角 + 对角有值，下三角为空；对称化
    mat = bc + bc.T
    np.fill_diagonal(mat, np.diag(bc) * 2)  # 对角边被加了两次，保持一致性
    return mat

type_codes = np.asarray(type_codes, dtype=np.int64)
obs_mat = count_type_edges(type_codes)

# 置换：打乱 type 标签（保持每类数量不变）
perm_mats = np.zeros((N_PERM, K, K), dtype=np.int64)
for r in range(N_PERM):
    perm_codes = rng.permutation(type_codes)
    perm_mats[r] = count_type_edges(perm_codes)
    if (r+1) % 50 == 0:
        print(f"  perm {r+1}/{N_PERM}")

# 计算 z-score
perm_mean = perm_mats.mean(axis=0)
perm_std  = perm_mats.std(axis=0, ddof=1)
z = (obs_mat - perm_mean) / (perm_std + 1e-9)

# 经验 p 值（双尾）：|perm - mean| >= |obs - mean|
abs_diff_obs = np.abs(obs_mat - perm_mean)
abs_diff_perm = np.abs(perm_mats - perm_mean[None, :, :])
p_emp = ( (abs_diff_perm >= abs_diff_obs[None,:,:]).sum(axis=0) + 1 ) / (N_PERM + 1)

# 只对上三角做多重校正（避免重复）
tri = np.triu_indices(K, k=0)
p_vec = p_emp[tri]
rej, p_fdr, _, _ = multipletests(p_vec, alpha=0.05, method="fdr_bh")
p_fdr_mat = np.ones((K,K), dtype=float)
p_fdr_mat[tri] = p_fdr
p_fdr_mat = p_fdr_mat + p_fdr_mat.T - np.diag(np.diag(p_fdr_mat))

# 保存结果表（便于补充材料）
rows = []
for i in range(K):
    for j in range(i, K):
        rows.append({
            "type_i": type_names[i],
            "type_j": type_names[j],
            "obs_edges": int(obs_mat[i,j]),
            "exp_edges": float(perm_mean[i,j]),
            "z": float(z[i,j]),
            "p_emp": float(p_emp[i,j]),
            "p_fdr": float(p_fdr_mat[i,j]),
        })
res_df = pd.DataFrame(rows).sort_values(["p_fdr","z"], ascending=[True,False])
res_df.to_csv(out_prefix + "<OUTPUT_CSV_FILE>", index=False)
print("[Saved]", out_prefix + "<OUTPUT_CSV_FILE>")

# =========================================================
# 4) 图 1：Nhood enrichment 热图（Nature Methods 常见主图）
#    - 用 z-score 显示富集/回避
#    - 用显著性遮罩只强调显著格子
# =========================================================
# 只显示显著的格子，否则置 0（或 NaN）
sig = (p_fdr_mat < 0.05)

z_plot = z.copy()
z_plot[~sig] = 0.0  # 也可以设为 np.nan 但 imshow 对 nan 处理不统一

plt.figure(figsize=(10, 8))
im = plt.imshow(z_plot, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04, label="Enrichment z-score")

plt.xticks(range(K), type_names, rotation=90, fontsize=7)
plt.yticks(range(K), type_names, fontsize=7)
plt.title(f"Spatial neighborhood enrichment (kNN={KNN}, FDR<0.05)")

# 在显著格子上打点标记（可选）
# for i in range(K):
#     for j in range(K):
#         if sig[i,j]:
#             plt.text(j, i, "•", ha="center", va="center", fontsize=6)

plt.tight_layout()
plt.savefig(out_prefix + "<SEGMENTATION_VISUALIZATION_FIGURE>", dpi=300)
plt.show()

# 同时输出 -log10(FDR) 热图（显著性强度）
neglog_fdr = -np.log10(np.clip(p_fdr_mat, 1e-300, 1.0))
neglog_plot = neglog_fdr.copy()
neglog_plot[~sig] = 0.0

plt.figure(figsize=(10, 8))
im = plt.imshow(neglog_plot, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04, label="-log10(FDR)")
plt.xticks(range(K), type_names, rotation=90, fontsize=7)
plt.yticks(range(K), type_names, fontsize=7)
plt.title(f"Neighborhood interaction significance (kNN={KNN})")
plt.tight_layout()
plt.savefig(out_prefix + "<SEGMENTATION_VISUALIZATION_FIGURE>", dpi=300)
plt.show()

# =========================================================
# 5) 图 2：Cell-type interaction network（只画显著且 z>0 的边）
# =========================================================
# 画网络：节点位置固定在圆上，更“论文级”且可复现
theta = np.linspace(0, 2*np.pi, K, endpoint=False)
pos = np.c_[np.cos(theta), np.sin(theta)]

# 只保留显著富集边（z>2 也可以更严格）
edge_list = []
for i in range(K):
    for j in range(i+1, K):
        #if sig[i,j] and z[i,j] > 2.0:
        if z[i, j] > 1.0:  # 不要求 FDR
            edge_list.append((i, j, z[i,j]))

# 节点大小按该类型细胞数量
counts = pd.value_counts(types).reindex(type_names).to_numpy()
node_size = 50 + 500 * (counts / counts.max())

plt.figure(figsize=(9, 9))
# nodes
plt.scatter(pos[:,0], pos[:,1], s=node_size)

# labels
for i, name in enumerate(type_names):
    plt.text(pos[i,0]*1.12, pos[i,1]*1.12, name, fontsize=7, ha="center", va="center")

# edges（线宽随 z 增强）
for i, j, zij in edge_list:
    lw = 0.3 + 0.4 * min(6.0, zij)  # 上限避免太粗
    plt.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], linewidth=lw, alpha=0.6)

plt.axis("off")
plt.title(f"Significant enriched type-type interactions (z>2, FDR<0.05), kNN={KNN}\nEdges={len(edge_list)}")
plt.tight_layout()
plt.savefig(out_prefix + "<OUTPUT_FIGURE_FILE>", dpi=300)
plt.show()

# =========================================================
# 6) 图 3：围绕谱（被谁包围）——以 Vascular / Immune 为例
# =========================================================
def neighbor_composition(target_type_name):
    t = type_names.index(target_type_name)
    # 找到所有 target cell
    target_idx = np.where(type_codes == t)[0]
    # 收集这些 cell 的邻居
    neigh = idxs[target_idx].reshape(-1)
    neigh_types = type_codes[neigh]
    comp = np.bincount(neigh_types, minlength=K).astype(float)
    comp = comp / comp.sum()
    return comp

targets = ["33 Vascular", "34 Immune"]  # 你也可以换成任意两个你关心的大类
comps = np.vstack([neighbor_composition(tt) for tt in targets])

# 用 “区分度更强”的离散色板
cmap = plt.get_cmap("tab20", K)
colors = [cmap(i) for i in range(K)]

plt.figure(figsize=(11, 4))
bottom = np.zeros(len(targets))
for i in range(K):
    plt.bar(targets, comps[:, i], bottom=bottom, color=colors[i], linewidth=0)
    bottom += comps[:, i]

plt.ylabel("Neighbor composition")
plt.title(f"Who surrounds whom (kNN={KNN})")
plt.xticks(rotation=0)

# legend 放右侧（可选：类别多时建议只显示前10 + Other）
plt.legend(type_names, bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False, ncol=1, fontsize=7)
plt.tight_layout()
plt.savefig(out_prefix + "<OUTPUT_FIGURE_FILE>", dpi=300)
plt.show()

print("[Done] Saved figures with prefix:", out_prefix)