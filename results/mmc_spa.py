import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from esda.moran import Moran
from libpysal.weights import WSP
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 数据准备
# =========================
levels = ["subclass", "supertype", "cluster"]

# 平均 bootstrapping confidence
our_conf = [0.6391, 0.7683, 0.6215]
bin8_conf = [0.5438, 0.7260, 0.5765]

# same-label neighbor ratio
our_same = [0.1218, 0.0880, 0.0560]
bin8_same = [0.1009, 0.0687, 0.0449]

# 你可以按需要改颜色
color_our = "#4C72B0"
color_bin8 = "#DD8452"

# 柱子宽度
width = 0.34
x = np.arange(len(levels))


# =========================
# 2. 作图函数：分组柱状图
# =========================
def plot_grouped_bar(levels, vals1, vals2, label1, label2, ylabel, title, ylim=None, save_path=None):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    bars1 = ax.bar(x - width/2, vals1, width, label=label1, color=color_our)
    bars2 = ax.bar(x + width/2, vals2, width, label=label2, color=color_bin8)

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    #ax.set_title(title, fontsize=13)
    ax.legend(frameon=False, fontsize=10)

    if ylim is not None:
        ax.set_ylim(ylim)

    # 去掉上右边框，更像论文图
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 在柱子上标数值
    def add_labels(bars):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width()/2,
                h + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()


# =========================
# 3. 图1：平均注释置信度
# =========================
plot_grouped_bar(
    levels=levels,
    vals1=our_conf,
    vals2=bin8_conf,
    label1="our",
    label2="8 μm bins",
    ylabel="Mean bootstrapping confidence",
    title="Fine-grained annotation confidence",
    ylim=(0.45, 0.82),
    save_path="<OUTPUT_FIGURE_FILE>"
)


# =========================
# 4. 图2：空间局部连续性
# =========================
plot_grouped_bar(
    levels=levels,
    vals1=our_same,
    vals2=bin8_same,
    label1="our",
    label2="8 μm bins",
    ylabel="Same-label neighbor ratio",
    title="Spatial coherence of assigned labels",
    ylim=(0.0, 0.14),
    save_path="<OUTPUT_FIGURE_FILE>"
)

exit()


# =========================================================
# 1. 文件路径
# =========================================================
our_adata_file = "<CELL_LEVEL_EXPRESSION_H5AD>"
our_csv_file = "<MAPMYCELLS_ANNOTATION_CSV>"

bin8_adata_file = "<CELL_LEVEL_EXPRESSION_H5AD>"
bin8_csv_file = "<MAPMYCELLS_ANNOTATION_CSV>"

# =========================================================
# 2. 读取数据并映射注释
# =========================================================
def load_adata_with_mmc(adata_file, mmc_file):
    adata = sc.read_h5ad(adata_file)
    mmc = pd.read_csv(mmc_file, comment="#")

    mmc["cell_id"] = mmc["cell_id"].astype(str)
    mmc = mmc.set_index("cell_id")

    cols = [
        "class_name", "subclass_name", "supertype_name", "cluster_name",
        "class_bootstrapping_probability", "subclass_bootstrapping_probability",
        "supertype_bootstrapping_probability", "cluster_bootstrapping_probability"
    ]

    for c in cols:
        adata.obs["mmc_" + c] = mmc.reindex(adata.obs_names)[c].values

    return adata

print("Loading our_seg...")
adata_our = load_adata_with_mmc(our_adata_file, our_csv_file)

print("Loading bin8...")
adata_bin8 = load_adata_with_mmc(bin8_adata_file, bin8_csv_file)

# =========================================================
# 3. 构建空间 kNN 图
# =========================================================
def build_knn_graph(coords, k=8):
    """
    coords: (n,2) numpy array
    return:
        neighbor_idx: 每个点的k个邻居索引（不含自己）
        W_sparse: scipy csr sparse weight matrix, row-standardized
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # 去掉第一个（自己）
    neighbor_idx = indices[:, 1:]

    n = coords.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = neighbor_idx.reshape(-1)
    data = np.ones(len(rows), dtype=np.float32)

    W = csr_matrix((data, (rows, cols)), shape=(n, n))

    # row-standardize
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv_row = 1.0 / row_sums
    D_inv = csr_matrix((inv_row, (np.arange(n), np.arange(n))), shape=(n, n))
    W_std = D_inv @ W

    return neighbor_idx, W_std

# =========================================================
# 4. same-label neighbor ratio
# =========================================================
def compute_same_label_ratio(labels, neighbor_idx):
    """
    labels: array-like, categorical labels
    neighbor_idx: (n, k)
    """
    labels = np.asarray(labels).astype(str)
    neighbor_labels = labels[neighbor_idx]   # (n, k)
    same = (neighbor_labels == labels[:, None])
    return same.mean()

# =========================================================
# 5. one-vs-rest weighted Moran's I
# =========================================================
def compute_weighted_moran_for_categorical(labels, W_sparse, min_count=20):
    """
    对 categorical labels：
    每个label构造二元向量(1=属于该label, 0=不属于)
    分别算 Moran's I，再按该label频数加权平均

    min_count: 出现太少的label不参与统计，避免极稀有label噪声过大
    """
    labels = pd.Series(labels).astype(str)
    counts = labels.value_counts()

    valid_labels = counts[counts >= min_count].index.tolist()

    if len(valid_labels) == 0:
        return np.nan, pd.DataFrame(columns=["label", "count", "moran_I"])

    wsp = WSP(W_sparse)

    results = []
    for lb in valid_labels:
        x = (labels.values == lb).astype(np.float32)
        try:
            mi = Moran(x, wsp, permutations=0)
            results.append([lb, counts[lb], mi.I])
        except Exception:
            continue

    if len(results) == 0:
        return np.nan, pd.DataFrame(columns=["label", "count", "moran_I"])

    df = pd.DataFrame(results, columns=["label", "count", "moran_I"])
    weighted_mean_I = np.average(df["moran_I"], weights=df["count"])

    return weighted_mean_I, df.sort_values("moran_I", ascending=False)

# =========================================================
# 6. 总分析函数
# =========================================================
def analyze_spatial_continuity(adata, name, levels, k=8, min_count=20):
    print("\n" + "=" * 90)
    print(f"Analyzing spatial continuity for: {name}")
    print("=" * 90)

    coords = adata.obs[["px", "py"]].to_numpy()
    print(f"n_objects = {coords.shape[0]}")

    print("Building kNN graph...")
    neighbor_idx, W_sparse = build_knn_graph(coords, k=k)

    all_results = {}

    for level in levels:
        label_col = "mmc_" + level

        # 去除缺失
        mask = ~adata.obs[label_col].isna().values
        sub_labels = adata.obs.loc[mask, label_col].astype(str).values
        sub_coords = coords[mask]

        print("\n" + "-" * 80)
        print(f"Level: {level}")
        print(f"valid objects: {len(sub_labels)}")

        # 这个层级单独重建 knn
        neighbor_idx_sub, W_sparse_sub = build_knn_graph(sub_coords, k=k)

        # 1) same-label neighbor ratio
        same_ratio = compute_same_label_ratio(sub_labels, neighbor_idx_sub)
        print(f"same-label neighbor ratio (k={k}): {same_ratio:.4f}")

        # 2) weighted Moran's I
        weighted_I, moran_df = compute_weighted_moran_for_categorical(
            sub_labels, W_sparse_sub, min_count=min_count
        )
        print(f"weighted one-vs-rest Moran's I (min_count>={min_count}): {weighted_I:.4f}")
        print(f"labels included in Moran aggregation: {moran_df.shape[0]}")

        all_results[level] = {
            "same_label_ratio": same_ratio,
            "weighted_moran_I": weighted_I,
            "moran_table": moran_df
        }

    return all_results

# =========================================================
# 7. 执行分析
# =========================================================
levels = ["subclass_name", "supertype_name", "cluster_name"]

res_our = analyze_spatial_continuity(
    adata_our,
    name="our_seg",
    levels=levels,
    k=8,
    min_count=20
)

res_bin8 = analyze_spatial_continuity(
    adata_bin8,
    name="bin8",
    levels=levels,
    k=8,
    min_count=20
)

# =========================================================
# 8. 汇总输出
# =========================================================
print("\n" + "#" * 100)
print("Summary")
print("#" * 100)

for level in levels:
    print(f"\n[{level}]")
    print(f"our_seg  | same-label ratio = {res_our[level]['same_label_ratio']:.4f} | weighted Moran's I = {res_our[level]['weighted_moran_I']:.4f}")
    print(f"bin8     | same-label ratio = {res_bin8[level]['same_label_ratio']:.4f} | weighted Moran's I = {res_bin8[level]['weighted_moran_I']:.4f}")