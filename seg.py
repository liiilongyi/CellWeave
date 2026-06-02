import numpy as np
import scanpy as sc
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree   # 用 KDTree 高效做最近邻

def summarize_seed_stats(adata, key):
    """
    输入：adata（AnnData）
    功能：统计 seed 列的基本情况
    包括：
        1. 非 -1 的数量
        2. 0 的数量
        3. 自然数（>0）的种类数
        4. 平均每个自然数出现次数
    """
    seed = adata.obs[key].to_numpy()

    # 2. 0 出现多少次
    count_zero = np.sum(seed == 0)
    # 3. 自然数（>0）有多少种不同标签
    valid_labels = seed[seed > 0]
    num_unique_labels = len(np.unique(valid_labels))
    # 4. 平均每个自然数出现多少次
    avg_per_label = len(valid_labels) / num_unique_labels if num_unique_labels > 0 else 0
    print("0 的数量 =", count_zero)
    print("自然数标签种类数 =", num_unique_labels)
    print("平均每个自然数出现次数 =", avg_per_label)

def visualize_msf_mask_with_seeds(adata, mask_key="cell_id_msf", seed_key="seed", figsize=(6,6), output_path=None, show=True):
    """
    根据 bin 坐标（从 barcode 解析），显示：
    - mask_key != 0 → 白色
    - mask_key == 0 → 黑色
    - seed_key > 0 → 红色（覆盖在上层）
    输出为 RGB 图像。
    """

    # ===== 1. 解析 barcode 中的坐标 =====
    idx_series = adata.obs_names.to_series()
    rc = idx_series.str.extract(r"s_002um_(\d+)_(\d+)-1")
    if rc.isnull().any().any():
        print("警告：barcode 解析部分失败！请检查格式。")

    rows = rc[0].astype(int).to_numpy()
    cols = rc[1].astype(int).to_numpy()

    # ===== 2. MSF mask 和 seed =====
    msf_labels = adata.obs[mask_key].to_numpy()
    seed_labels = adata.obs[seed_key].to_numpy()

    # ===== 3. 起始坐标平移为 0-based =====
    row_min, col_min = rows.min(), cols.min()
    rows0 = rows - row_min
    cols0 = cols - col_min

    H = rows0.max() + 1
    W = cols0.max() + 1
    print(f"掩码尺寸：H={H}, W={W}")

    # ===== 4. 初始化 RGB 掩码图（黑色背景） =====
    mask_img = np.zeros((H, W, 3), dtype=np.uint8)

    # ===== 5. 绘制 MSF 掩码（白色） =====
    white_mask = (msf_labels != 0)
    mask_img[rows0[white_mask], cols0[white_mask]] = [255, 255, 255]

    # ===== 6. 绘制 Seed 掩码（红色） =====
    seed_mask = (seed_labels > 0)
    mask_img[rows0[seed_mask], cols0[seed_mask]] = [255, 0, 0]

    # ===== 7. 显示 / 保存 =====
    plt.figure(figsize=figsize)
    plt.imshow(mask_img, origin="upper")
    plt.axis("off")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", dpi=800)
        print(f"掩码已保存到: {output_path}")

    plt.show()

def build_voronoi_from_seed(adata, seed_key="seed", use_barcode=True):
    """
    根据 adata.obs[seed_key] 和 bin 坐标（从 barcode 解析）构建离散 Voronoi：
    - 每个 seed (>0) 有一个质心
    - 每个 bin 归属最近的 seed 质心
    结果写入 adata.obs["seed_voronoi"]，表示“理论势力范围”。

    返回：
        voronoi_owner: np.ndarray，长度 = n_obs，每个元素是对应的 seed_id（或 0 表示无 seed）
    """

    # ===== 1. 解析每个 bin 的坐标 (row, col) =====
    idx_series = adata.obs_names.to_series()

    # 假设 barcode 格式类似：s_002um_00014_00074-1
    rc = idx_series.str.extract(r"s_002um_(\d+)_(\d+)-1")
    if rc.isnull().any().any():
        print("⚠️ 警告：有 barcode 解析失败，请检查正则表达式/格式。")

    rows = rc[0].astype(int).to_numpy()
    cols = rc[1].astype(int).to_numpy()

    # 平移到从 0 开始（方便后面可视化用）
    row_min, col_min = rows.min(), cols.min()
    rows0 = rows - row_min
    cols0 = cols - col_min

    # ===== 2. 读取 seed 标签 =====
    seeds = adata.obs[seed_key].to_numpy().astype(np.int64)
    # 把 <0 的都当作未标记
    seeds[seeds < 0] = 0

    # 只保留有核的 bin（seed > 0）
    has_seed_mask = seeds > 0
    if has_seed_mask.sum() == 0:
        raise ValueError("seed 中没有 >0 的核标签，无法构建 Voronoi。")

    seed_ids = seeds[has_seed_mask]
    seed_rows = rows0[has_seed_mask]
    seed_cols = cols0[has_seed_mask]

    # ===== 3. 计算每个 seed 的质心 =====
    df_seed = pd.DataFrame({
        "seed": seed_ids,
        "row": seed_rows,
        "col": seed_cols,
    })

    # groupby 每个 seed，算平均坐标作为质心
    centroids = df_seed.groupby("seed")[["row", "col"]].mean()
    centroid_coords = centroids[["row", "col"]].to_numpy()   # (K, 2)
    centroid_seeds = centroids.index.to_numpy()              # 对应的 seed id

    print(f"共有 {len(centroid_seeds)} 个 seed，有质心坐标可用于 Voronoi。")

    # ===== 4. 对所有 bin 做最近质心分配（离散 Voronoi） =====
    all_points = np.column_stack([rows0, cols0])   # (N, 2)
    tree = cKDTree(centroid_coords)
    dist, idx = tree.query(all_points)  # idx: 每个点对应最近质心的索引
    voronoi_owner = np.zeros(len(adata), dtype=np.int64)
    voronoi_owner[:] = 0      # 默认 0 表示“不属于任何 seed（理论上不会发生）”
    voronoi_owner = centroid_seeds[idx]  # 用最近质心对应的 seed id 填充

    # ===== 5. 写回 adata.obs =====
    adata.obs["seed_voronoi"] = voronoi_owner
    print("✅ 已在 adata.obs['seed_voronoi'] 中写入 Voronoi 归属 seed。")
    return adata

data = np.load("<EDGE_PROBABILITY_NPZ>", allow_pickle=True)
#data = np.load("<EDGE_PROBABILITY_NPZ>", allow_pickle=True)
#data = np.load("<EDGE_PROBABILITY_NPZ>", allow_pickle=True)
#data = np.load("<EDGE_PROBABILITY_NPZ>", allow_pickle=True)
#data = np.load("<EDGE_PROBABILITY_NPZ>", allow_pickle=True)


adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
#adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
#adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")

adata.var['gene'] = adata.var['gene'].astype(str).str.upper()
nnz_per_bin = np.diff(adata.X.indptr)

# 找出非全零的 bin
keep = nnz_per_bin > 0
adata = adata[keep].copy()
with open("<SCGPT_VOCAB_JSON>", "r") as f:
    vocab = json.load(f)
vocab_genes = {k for k in vocab.keys() if not k.startswith("<")}
genes = adata.var["gene"].astype(str)
keep_gene_mask = genes.isin(vocab_genes)
adata = adata[:, keep_gene_mask].copy()
# 3. 在这个子矩阵上再删一次全 0 行
X = adata.X
cell_sum = np.array(X.sum(axis=1)).ravel()
nonzero_cell_mask = cell_sum > 0
adata = adata[nonzero_cell_mask].copy()

# print(adata)

summarize_seed_stats(adata, 'seed')

adata = build_voronoi_from_seed(adata, seed_key="seed")


#print(data.files)
edge_index = data['edge_index']    # shape (2, E)
alpha = data['alpha']              # shape (E,)
print("edge_index:", edge_index.shape)
print("alpha:", alpha.shape, alpha.min(), alpha.max())

edge_index = data["edge_index"].astype(np.int64)  # shape (2, E)
alpha = data["alpha"].astype(np.float32)          # shape (E, )

# —— 如果 edge_index 是 1-based，把它改成 0-based ——
if edge_index.min() == 1:
    edge_index -= 1

# ====== 节点标签（核分割 seeds） ======
# 假设图的节点顺序与 adata.obs 顺序一致
seeds = adata.obs["seed"].to_numpy().astype(np.int64)  # shape (N,)
seeds[seeds < 0] = 0
voronoi_owner = adata.obs["seed_voronoi"].to_numpy().astype(np.int64)

# 坐标（用 px/py 来度量距离）
px = adata.obs["px"].to_numpy().astype(np.float32)
py = adata.obs["py"].to_numpy().astype(np.float32)

N = seeds.shape[0]
E = edge_index.shape[1]

print(f"N = {N}, E = {E}, seeds > 0 = {(seeds > 0).sum()}")

# ===== 1. 计算每个核的质心和“核半径” =====
nuc_mask = seeds > 0
df_nuc = pd.DataFrame({
    "seed": seeds[nuc_mask],
    "px": px[nuc_mask],
    "py": py[nuc_mask],
})

# 1.1 每个 seed 的质心
centroids = df_nuc.groupby("seed")[["px", "py"]].mean()

# 1.2 把质心映射回核内每个 bin，算到质心的距离
df_nuc = df_nuc.join(centroids, on="seed", rsuffix="_c")
dx = df_nuc["px"] - df_nuc["px_c"]
dy = df_nuc["py"] - df_nuc["py_c"]
df_nuc["dist"] = np.sqrt(dx*dx + dy*dy)

# 1.3 每个 seed 的“核半径”：用 90% 分位数（比最大值稳健）
r_nuc = df_nuc.groupby("seed")["dist"].quantile(0.9)

# ===== 1.4 把质心和半径变成数组，方便用下标快速索引 =====
max_seed_id = max(int(centroids.index.max()), int(voronoi_owner.max()))

center_px = np.zeros(max_seed_id + 1, dtype=np.float32)
center_py = np.zeros(max_seed_id + 1, dtype=np.float32)
radius_nuc = np.zeros(max_seed_id + 1, dtype=np.float32)

center_px[centroids.index.to_numpy()] = centroids["px"].to_numpy()
center_py[centroids.index.to_numpy()] = centroids["py"].to_numpy()
radius_nuc[r_nuc.index.to_numpy()] = r_nuc.to_numpy()

# 1.5 允许的最大细胞半径：k 倍核半径
k_radius = 2.5   # 可以先试 3.0，再看平均 cell 大小调整到 2.5/4.0 等       # 原 2.5
radius_limit = k_radius * radius_nuc

# 1.6 给每个 bin 算“到自己 Voronoi 核质心的距离”和“该核允许的最大半径”
owner = voronoi_owner  # 每个 bin 对应的 voronoi seed
cx_all = center_px[owner]
cy_all = center_py[owner]
dx_all = px - cx_all
dy_all = py - cy_all
dist_to_center = np.sqrt(dx_all*dx_all + dy_all*dy_all)  # 每个 bin 到自己核心的距离

radius_limit_for_node = radius_limit[owner]  # 每个 bin 对应的“允许半径”

# ========= 2. 准备边（过滤掉越界的，以防万一） =========
u = edge_index[0]
v = edge_index[1]
valid_mask = (u >= 0) & (u < N) & (v >= 0) & (v < N)
u = u[valid_mask]
v = v[valid_mask]
alpha = alpha[valid_mask]

# 再按 alpha 做阈值过滤，让图“更稀疏”，细胞别扩散太远
alpha_thresh = 0.3   # 可以先试 0.90 / 0.92 / 0.95 调一调


strong_mask = alpha >= alpha_thresh
u = u[strong_mask]
v = v[strong_mask]
alpha = alpha[strong_mask]

E_valid = u.shape[0]
print(f"Edges after threshold (alpha >= {alpha_thresh}): {E_valid}")

same_voronoi = voronoi_owner[u] == voronoi_owner[v]
u = u[same_voronoi]
v = v[same_voronoi]
alpha = alpha[same_voronoi]
E_valid = u.shape[0]
print(f"Edges after Voronoi filtering: {E_valid}")

# 2.3 距离约束：两端都不能超过自己的允许半径
#   dist_to_center: 每个节点到自己 voronoi 核质心的距离
#   radius_limit_for_node: 每个节点对应核的最大允许半径 = k * r_nuc
dist_ok = (dist_to_center[u] <= radius_limit_for_node[u]) & \
          (dist_to_center[v] <= radius_limit_for_node[v])

u = u[dist_ok]
v = v[dist_ok]
alpha = alpha[dist_ok]

E_valid = u.shape[0]
print(f"Edges after distance filtering (k_radius={k_radius}): {E_valid}")


# 把 alpha 转成“代价”：cost 越小 => 越相似 => 越应该先连在一起
cost = 1.0 - alpha  # 或者用 -alpha，然后排序时反向；这里更直观


# ========= 3. 并查集（Union-Find）结构 =========
class UnionFind:
    def __init__(self, n, labels):
        # parent：每个节点的代表元
        self.parent = np.arange(n, dtype=np.int64)
        # rank：按秩合并
        self.rank = np.zeros(n, dtype=np.int8)
        # comp_label：每个连通分量的标签（0 = 无标签；>0 = 某个 seed）
        self.comp_label = labels.copy()

    def find(self, x):
        # 路径压缩
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_seeded(self, x, y):
        """
        带种子规则的合并逻辑：
        - 如果两个分量标签都是 0：正常合并，新分量 label 仍为 0。
        - 如果一个为 0，另一个 >0：合并，新分量 label = 非零那边。
        - 如果两个都 >0 且相同：合并，label 保持。
        - 如果两个都 >0 且不同：不合并（形成边界）。
        """
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return

        lx = self.comp_label[px]
        ly = self.comp_label[py]

        # 两边标签都 >0 且不同 -> 不合并，保留为边界
        if lx > 0 and ly > 0 and lx != ly:
            return

        # 按秩合并
        if self.rank[px] < self.rank[py]:
            px, py = py, px  # 保证 px rank >= py rank

        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        # 更新新根的 comp_label
        lx = self.comp_label[px]
        ly = self.comp_label[py]
        if lx == 0 and ly == 0:
            self.comp_label[px] = 0
        elif lx == 0 and ly > 0:
            self.comp_label[px] = ly
        elif lx > 0 and ly == 0:
            self.comp_label[px] = lx
        else:  # lx == ly > 0
            self.comp_label[px] = lx


# 初始化并查集：每个节点单独成分，标签为 seeds
uf = UnionFind(N, seeds)

# ========= 4. 按代价排序边，并依次尝试合并 =========
order = np.argsort(cost)  # cost 小的（alpha 大的）优先
u_sorted = u[order]
v_sorted = v[order]

print("Start seeded MSF growth (this may take a while)...")

for i in tqdm(range(E_valid), desc="seed MSF growth"):
    uf.union_seeded(u_sorted[i], v_sorted[i])
    # 可以偶尔打印进度（可选）
    # if (i + 1) % 1000000 == 0:
    #     print(f"Processed {i+1}/{E_valid} edges")

print("MSF growth done.")


# ========= 5. 得到每个节点的最终标签 =========
final_labels = np.zeros(N, dtype=np.int64)
for i in range(N):
    root = uf.find(i)
    final_labels[i] = uf.comp_label[root]

# 统计一下
print("Final labels stats:")
print("  total nodes:", N)
print("  unlabeled (0):", np.sum(final_labels == 0))
print("  unique labels (excluding 0):", len(np.unique(final_labels[final_labels > 0])))

# ========= 6. 写回到 adata，并保存 =========
adata.obs["msf"] = final_labels

summarize_seed_stats(adata, "msf")
visualize_msf_mask_with_seeds(adata, mask_key="msf", seed_key="seed", output_path="<SEGMENTATION_VISUALIZATION_FIGURE>")  # 需改

# 如果你的图只覆盖了 in_tissue 或者一个子集，需要：
#   adata.obs["cell_id_msf"][:] = 0
#   adata.obs.loc[mapping_graphnode_to_adata_index, "cell_id_msf"] = final_labels_for_graphnodes


adata2 = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")       # 需改

adata2.obs["<OUTPUT_SEGMENTATION_COLUMN>"] = final_labels     # 需改

adata2.write_h5ad("<SEGMENTED_BIN_H5AD>")           # 需改
print("Saved!")

print(adata2)
