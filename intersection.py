import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import pearsonr
from collections import defaultdict
from tqdm.auto import tqdm

adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
print(adata)

if not sp.isspmatrix_csr(adata.X):
    adata.X = adata.X.tocsr()

X = adata.X  # (n_bins, n_genes)

# 两种分割方法的 cell id
msf = adata.obs["<FIRST_SEGMENTATION_COLUMN>"].to_numpy().astype(np.int64)
stp = adata.obs["b2c"].to_numpy().astype(np.int64)

labels = ["our", "Bin2cell"]
save_path = '<OUTPUT_FIGURE_FILE>'

n_bins = adata.n_obs
n_genes = adata.n_vars
print(f"n_bins: {n_bins}, n_genes: {n_genes}")

# =========================================================
# 1. 构建 msf × STP 的重叠计数矩阵，找到每个 msf cell 对应的“重叠最多的 STP cell”
# =========================================================

# 有效 bin：两个方法都给了有效 cell id（>0 且 STP != -1）
valid = (msf > 0) & (stp > 0) & (stp != -1)
msf_valid = msf[valid]
stp_valid = stp[valid]

print(f"valid bins: {valid.sum()} / {n_bins}")

# 把各自的 cell id 压缩成 0..n_unique-1 的索引
unique_msf, inv_msf = np.unique(msf_valid, return_inverse=True)
unique_stp, inv_stp = np.unique(stp_valid, return_inverse=True)

print("unique the first method cells:", len(unique_msf))
print("unique the second method cells:", len(unique_stp))

# 用稀疏矩阵统计重叠次数
overlap = sp.coo_matrix(
    (np.ones_like(inv_msf, dtype=np.int32), (inv_msf, inv_stp)),
    shape=(len(unique_msf), len(unique_stp))
).tocsr()

msf_to_best_stp = {}  # msf_cell_id -> best stp_cell_id
msf_best_overlap = {}  # msf_cell_id -> overlap bin count

for i, cid in enumerate(unique_msf):
    row = overlap.getrow(i)
    if row.nnz == 0:
        # 这个 msf cell 和任何 STP cell 都没有重叠
        continue
    j = row.indices[row.data.argmax()]
    msf_to_best_stp[cid] = unique_stp[j]
    msf_best_overlap[cid] = int(row.data.max())

print(f"the first method cells with any overlap: {len(msf_to_best_stp)}")

# 可以简单看几个例子
print("示例映射（前 10 个）：")
for k in list(msf_to_best_stp.keys())[:10]:
    print(f"the first method cell {k}  <->  the second method cell {msf_to_best_stp[k]}, overlap bins = {msf_best_overlap[k]}")

# =========================================================
# 2. 为两个分割方法构建 “cell_id -> bin 索引列表” 映射
# =========================================================

msf2bins = defaultdict(list)
stp2bins = defaultdict(list)

print("构建 cell_id -> bin 索引映射（the first method bins /the second method bins）...")
for idx in tqdm(range(n_bins), desc="Indexing bins to cells"):
    cid_m = msf[idx]
    cid_s = stp[idx]
    if cid_m > 0:
        msf2bins[cid_m].append(idx)
    if cid_s > 0 and cid_s != -1:
        stp2bins[cid_s].append(idx)

# 转成 numpy 数组
msf2bins = {cid: np.array(idxs, dtype=np.int64) for cid, idxs in msf2bins.items()}
stp2bins = {cid: np.array(idxs, dtype=np.int64) for cid, idxs in stp2bins.items()}

print("the first method bins cells:", len(msf2bins), "the second method bins cells:", len(stp2bins))

# 覆盖率简单看一下
all_msf_cells = set(msf2bins.keys())
msf_cells_with_overlap = set(msf_to_best_stp.keys())
msf_cells_no_overlap = all_msf_cells - msf_cells_with_overlap

print(f"总 the first method cells: {len(all_msf_cells)}")
print(f"有与 the second method 重叠的 the first method cells: {len(msf_cells_with_overlap)}")
print(f"完全没有与 the second method 重叠的 the first method cells: {len(msf_cells_no_overlap)}")


# =========================================================
# 3. 定义相关性计算函数
# =========================================================

def safe_pearson(a, b):
    """避免全 0 或常数向量导致 pearson 报错或 NaN。"""
    if np.all(a == 0) or np.all(b == 0):
        return np.nan
    return pearsonr(a, b)[0]
def compute_scores_for_msf_cell(cid_msf, X_csr):
    """
    对单个 msf cell = cid_msf：
      1) 找到它最匹配的 stp cell = cid_stp
      2) 计算三个区域的表达向量：x_int, x_c, x_c'
      3) 返回 corr(x_c, x_int), corr(x_c', x_int) 以及一些计数信息
    """
    if cid_msf not in msf_to_best_stp:
        # 没有任何重叠
        return None

    cid_stp = msf_to_best_stp[cid_msf]
    bins_m = msf2bins.get(cid_msf, None)
    bins_s = stp2bins.get(cid_stp, None)
    if bins_m is None or bins_s is None:
        return None

    # 三个区域：交集 / msf 独有 / stp 独有
    inter = np.intersect1d(bins_m, bins_s, assume_unique=False)
    if inter.size == 0:
        return None

    msf_only = np.setdiff1d(bins_m, inter, assume_unique=False)
    stp_only = np.setdiff1d(bins_s, inter, assume_unique=False)

    # 区域表达向量（按 bin 求和）
    x_int = X_csr[inter].sum(axis=0).A1
    x_c = X_csr[msf_only].sum(axis=0).A1 if msf_only.size > 0 else np.zeros(X_csr.shape[1])
    x_cp = X_csr[stp_only].sum(axis=0).A1 if stp_only.size > 0 else np.zeros(X_csr.shape[1])

    corr_m = safe_pearson(x_c, x_int)  # corr(x_c, x_int)
    corr_s = safe_pearson(x_cp, x_int)  # corr(x_c', x_int)

    return {
        "cid_msf": cid_msf,
        "cid_stp": cid_stp,
        "n_bins_msf": int(bins_m.size),
        "n_bins_stp": int(bins_s.size),
        "n_bins_inter": int(inter.size),
        "n_bins_msf_only": int(msf_only.size),
        "n_bins_stp_only": int(stp_only.size),
        "corr_msf_vs_int": corr_m,
        "corr_stp_vs_int": corr_s,
        "overlap_count": msf_best_overlap.get(cid_msf, 0),
    }


# =========================================================
# 4. 对所有有重叠的 msf cell 计算分数
# =========================================================

results = []
msf_cells_to_eval = list(msf_to_best_stp.keys())

print(f"开始对 {len(msf_cells_to_eval)} 个有重叠的 the first method cell 计算分数...")
for cid_msf in tqdm(msf_cells_to_eval, desc="Computing scores per msf cell"):
    res = compute_scores_for_msf_cell(cid_msf, X)
    if res is not None:
        results.append(res)

df_scores = pd.DataFrame(results)
print("评分结果前几行：")
print(df_scores.head())

print(f"有效评分的 the first method cells 数量: {df_scores.shape[0]}")

# 保存结果
df_scores.to_csv("<METRICS_OUTPUT_CSV>", index=False)
#print("<METRICS_OUTPUT_CSV>")

# 也可以挂到 adata.uns 里，方便后面继续用
adata.uns["msf_stp_overlap_scores"] = df_scores
#adata.write("<SEGMENTED_BIN_H5AD>")
#print("<SEGMENTED_BIN_H5AD>")


# =========================================================
# 5. 计算两个方法的全局分数（一个 msf，一个 STP）
# =========================================================

# 去掉 NaN 的行
valid_msf = ~np.isnan(df_scores["corr_msf_vs_int"].to_numpy())
valid_stp = ~np.isnan(df_scores["corr_stp_vs_int"].to_numpy())

# 用交集 bin 数作为权重
w = df_scores["n_bins_inter"].to_numpy().astype(float)

corr_msf = df_scores["corr_msf_vs_int"].to_numpy()
corr_stp = df_scores["corr_stp_vs_int"].to_numpy()

# 只在各自有值的位置做加权平均
w_msf = w[valid_msf]
w_stp = w[valid_stp]
corr_msf_valid = corr_msf[valid_msf]
corr_stp_valid = corr_stp[valid_stp]

# 避免除以 0
if w_msf.sum() > 0:
    score_msf = (w_msf * corr_msf_valid).sum() / w_msf.sum()
else:
    score_msf = np.nan

if w_stp.sum() > 0:
    score_stp = (w_stp * corr_stp_valid).sum() / w_stp.sum()
else:
    score_stp = np.nan

print("\n===== Global scores =====")
print(f"The first method global score (weighted corr):  {score_msf:.4f}")
print(f"The second method global score (weighted corr):  {score_stp:.4f}")


import matplotlib.pyplot as plt

plt.scatter(df_scores["corr_msf_vs_int"], df_scores["corr_stp_vs_int"], s=5, alpha=0.3)
plt.axline((0,0),(1,1), linestyle="--")
plt.xlabel("corr(msf_only, intersection)")
plt.ylabel("corr(stp_only, intersection)")
plt.title("Per-cell expression consistency comparison")
plt.show()




from scipy.stats import wilcoxon

df = df_scores.copy()

# 只取两边都有值的配对样本
pair = df[["corr_msf_vs_int", "corr_stp_vs_int"]].dropna()
w = df.loc[pair.index, "n_bins_inter"].to_numpy().astype(int)


p = wilcoxon(
    pair["corr_msf_vs_int"],
    pair["corr_stp_vs_int"]
).pvalue

msf = pair["corr_msf_vs_int"].to_numpy()
stp = pair["corr_stp_vs_int"].to_numpy()

msf_w = np.repeat(msf, w)
stp_w = np.repeat(stp, w)

data = [msf_w, stp_w]

plt.figure(figsize=(3,4))
bp = plt.boxplot(
    data,
    labels=labels,
    showfliers=False,
    patch_artist=True,
    widths=0.3
)

bp["boxes"][0].set_facecolor("#E07B7B")
bp["boxes"][1].set_facecolor("#4C9AA2")

plt.ylabel("Correlation")
plt.title("Expression consistency")

# =========================
# 关键优化 1：自适应 y 轴
# =========================
all_vals = np.concatenate(data)
y_min = np.nanpercentile(all_vals, 1)
y_max = np.nanpercentile(all_vals, 99)

padding = 0.15 * (y_max - y_min)
y_lower = max(0, y_min - padding)
y_upper = y_max + 2 * padding   # 预留 P-value 空间

plt.ylim(y_lower, y_upper)

# =========================
# 关键优化 2：P 值括号画在图内
# =========================
x1, x2 = 1, 2
y_bracket = y_max + 0.6 * padding
h = 0.15 * padding

plt.plot(
    [x1, x1, x2, x2],
    [y_bracket, y_bracket + h, y_bracket + h, y_bracket],
    lw=1.2,
    c="k"
)

plt.text(
    (x1 + x2) / 2,
    y_bracket + h * 1.2,
    f"P = {p:.2e}",
    ha="center",
    va="bottom"
)

plt.tight_layout()
plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600)
plt.show()
