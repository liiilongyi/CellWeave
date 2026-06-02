import scanpy as sc
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# 读取 h5ad 文件
adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
print(adata.obs['seed'].nunique())
# 1. 取出 barcode 和 seed
barcodes = adata.obs.index.astype(str).values  # index 是 barcode
seeds = adata.obs["seed"].to_numpy()          # int32，长度 = n_bins

n_bins = len(barcodes)
print("Total bins:", n_bins)

# 2. 从 barcode 中解析 (row, col)，如 s_002um_00014_00074-1 → 14, 74
rows = np.zeros(n_bins, dtype=int)
cols = np.zeros(n_bins, dtype=int)

# 根据你的 barcode 格式写正则
pattern = re.compile(r"s_002um_(\d+)_(\d+)-1")

for i, b in enumerate(barcodes):
    m = pattern.match(b)
    if m is None:
        raise ValueError(f"Unexpected barcode format: {b}")
    rows[i] = int(m.group(1))  # 00014 → 14
    cols[i] = int(m.group(2))  # 00074 → 74

print("row range:", rows.min(), rows.max())
print("col range:", cols.min(), cols.max())

# 3. 建一个 (row, col) → index 的查找表，方便找8邻域
rc2idx = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}

# 4. 遍历每个 bin，找 8 邻域，并根据 seed 生成带标签的边
edges = []   # 每一项：[row1, col1, row2, col2, label]

# 8 邻域的偏移（不含 0,0）
OFFS = [(-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)]

for i in tqdm(range(n_bins), desc="Building labeled edges"):
    seed_i = seeds[i]
    if seed_i == -1:
        # 没有对应核，直接跳过这个 bin
        continue

    r_i, c_i = rows[i], cols[i]

    for dr, dc in OFFS:
        r_j = r_i + dr
        c_j = c_i + dc
        key_j = (r_j, c_j)

        # 邻居是否存在
        j = rc2idx.get(key_j, None)
        if j is None:
            continue

        seed_j = seeds[j]
        if seed_j == -1:
            # 邻居也没有核，不考虑这条边
            continue

        # 为了避免 (i,j) 和 (j,i) 重复，只保留 i < j 的情况
        if i >= j:
            continue

        # 同 seed → 1，不同 seed → -1
        if seed_i == seed_j:
            label = 1
        else:
            label = -1

        edges.append([r_i, c_i, r_j, c_j, label])

# 5. 存成 DataFrame（更直观方便）
edge_df = pd.DataFrame(edges, columns=["row1", "col1", "row2", "col2", "label"])

print("Total edges with labels:", len(edge_df))
print(edge_df.head())
print("Positive labels (1):", (edge_df["label"] == 1).sum())
print("Negative labels (-1):", (edge_df["label"] == -1).sum())

# 你可以保存到文件里
edge_df.to_csv("<EDGE_LABELS_CSV>", index=False)
# 或者保存成 npz / pkl
# np.savez("<INPUT_OR_OUTPUT_NPZ>", edges=np.array(edges, dtype=int))

