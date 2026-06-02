import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from tqdm import tqdm

#path = "<NUCLEUS_SEGMENTATION_MASK_NPY>"
#path = "<NUCLEUS_SEGMENTATION_MASK_NPY>"
#path = "<NUCLEUS_SEGMENTATION_MASK_NPY>"    #(6000, 5803)
#path = "<NUCLEUS_SEGMENTATION_MASK_NPY>"
path = "<NUCLEUS_SEGMENTATION_MASK_NPY>"   # cellpose Ovarian
#path = "<NUCLEUS_SEGMENTATION_MASK_NPY>"
#path = "<NUCLEUS_SEGMENTATION_MASK_NPY>"


# Cellpose
x = np.load(path, allow_pickle=True).item()  # 关键：.item() 取出 dict
nuclei_mask = x["masks"]   # uint16, (H, W)


#nuclei_mask = np.load(path) # Stardist
print(nuclei_mask.shape)

#adata = sc.read_h5ad("<BIN_EXPRESSION_H5AD>")
adata = sc.read_h5ad("<BIN_EXPRESSION_H5AD>")


H, W = nuclei_mask.shape

# 像素坐标缩放因子
scale = 1  ### 需修改 ！！！

# spot 直径
diameter = 16
radius = diameter // 2      # 3

# 预先分配 seed 数组，默认 -1 表示“未匹配到明显细胞核”
seeds = np.full(adata.n_obs, -1, dtype=np.int32)

px_all = adata.obs['px'].to_numpy()
py_all = adata.obs['py'].to_numpy()

print(f"nuclei_mask shape (H,W): {H}, {W}")
print(f"每个 bin 理论上的 patch 大小（中间区域）：{(2*radius+1)} × {(2*radius+1)} = {(2*radius+1)**2} 像素")

# 2. 遍历每个 bin
for i in tqdm(range(adata.n_obs), desc="Assigning seeds"):
    # (1) 坐标缩放并取整
    x_pix = int(round(px_all[i] * scale))
    y_pix = int(round(py_all[i] * scale))

    # (2) 边界检查，保证不会越界
    x0 = max(0, x_pix - radius)
    x1 = min(W, x_pix + radius + 1)
    y0 = max(0, y_pix - radius)
    y1 = min(H, y_pix + radius + 1)

    patch = nuclei_mask[y0:y1, x0:x1]

    # 实际 patch 像素数（边缘处会比 7×7 小）
    n_pix = patch.size
    if n_pix == 0:
        continue

    # (3) 统计 label 次数（建议把 0 当作背景丢掉）
    vals = patch.ravel()
    vals = vals[vals > 0]       # 去掉背景 0，如果你想考虑背景也可删掉这行
    if vals.size == 0:
        continue

    labels, counts = np.unique(vals, return_counts=True)
    best_label = labels[np.argmax(counts)]
    best_count = counts.max()

    # (4) 若出现次数超过一半像素数，则认为这个核对应这个 bin
    if best_count > n_pix * 0.5:
        seeds[i] = best_label

# 3. 写入 adata.obs['seed']
adata.obs['seed'] = seeds

# 简单统计一下
n_assigned = np.sum(seeds != -1)
print(f"总 bin 数: {adata.n_obs}")
print(f"被分配到某个 nuclei label 作为 seed 的 bin 数: {n_assigned}")
print(f"比例: {n_assigned / adata.n_obs * 100:.2f}%")

print(adata.obs['seed'])
adata.write("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
print("done")

exit()


plt.figure(figsize=(6, 6))
plt.imshow(nuclei_mask, cmap="nipy_spectral")
plt.title("Nuclei mask (Cellpose)")
plt.axis("off")
plt.show()
exit()


# 读取核掩码
nuclei_mask = np.load(path)

print("Loaded nuclei mask.")
print("Shape:", nuclei_mask.shape)
print("dtype:", nuclei_mask.dtype)

# 查看最大 label（即核的数量）
num_nuclei = nuclei_mask.max()
print("Number of nuclei (labels):", num_nuclei)
