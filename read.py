import scanpy as sc
import re
import numpy as np
import matplotlib.pyplot as plt

# 读取 h5ad 文件
adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
print(adata)
exit()

# 1. 从 adata.obs 里拿出 barcode 和 seed
barcodes = adata.obs.index.astype(str).values   # index 是 barcode
seeds    = adata.obs["seed"].to_numpy()         # 长度 = n_bins

# 2. 解析 barcode，提取 array_row 和 array_col（自然数）
rows = []
cols = []

pattern = re.compile(r"s_002um_(\d+)_(\d+)-1")   # 适配形如 s_002um_00014_00074-1

for b in barcodes:
    m = pattern.match(b)
    if m is None:
        raise ValueError(f"Unexpected barcode format: {b}")
    r = int(m.group(1))   # 00014 -> 14
    c = int(m.group(2))   # 00074 -> 74
    rows.append(r)
    cols.append(c)

rows = np.array(rows, dtype=int)
cols = np.array(cols, dtype=int)

print("row range:", rows.min(), rows.max())
print("col range:", cols.min(), cols.max())

# 3. 区分 seed = -1 和 seed != -1
mask_seed = seeds != -1   # True 表示有核的 bin

# 4. 画图（背景黑色）
plt.figure(figsize=(8, 8))
ax = plt.gca()

# 背景设为黑色
ax.set_facecolor("black")
plt.gcf().patch.set_facecolor("black")

# 先画所有 bin（灰色）
plt.scatter(
    cols, rows,
    s=2,           # 点小一点
    c="dimgray",
    alpha=0.3,
    marker="s",      # 小方块，看起来更像网格
    linewidths=0
)

# 再画 seed != -1 的 bin（亮色，比如青色）
plt.scatter(
    cols[mask_seed], rows[mask_seed],
    s=4,
    c="cyan",
    alpha=0.9,
    marker="s",
    linewidths=0
)

plt.gca().invert_yaxis()     # 让 row 方向和图像方向一致（可选）
plt.axis("equal")
plt.axis("off")
plt.show()
