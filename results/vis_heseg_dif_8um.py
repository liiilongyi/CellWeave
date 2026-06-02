import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap
import os

# ===== 关闭 PIL 大图限制 =====
Image.MAX_IMAGE_PIXELS = None

# 图像路径
img_path = "<H_AND_E_IMAGE_FILE>"
# img_path = "<H_AND_E_IMAGE_FILE>"

# 新的 Parquet 路径
seg_path = "<GROUND_TRUTH_CELL_LABELS_PARQUET>"
# seg_path = "<GROUND_TRUTH_CELL_LABELS_PARQUET>"

# 修改为你的 cell ID 列名
mask_key = "cell_id"

# 输出路径调整到 mouse_brain 对应目录
output_path = "<SEGMENTATION_VISUALIZATION_FIGURE>"
out_dir = "<OUTPUT_DIRECTORY_OR_PREFIX>"

# =========================================================
# 1. 读数据 (从 Parquet)
# =========================================================
print("Loading parquet data...")
df = pd.read_parquet(seg_path)

# 获取物理像素坐标 (X=列, Y=行)
# Visium HD 官方标准列名是 pxl_col_in_fullres 和 pxl_row_in_fullres
# 如果你的表格中已经有 px, py 则优先使用 px, py
if "px" in df.columns and "py" in df.columns:
    px = df["px"].to_numpy(dtype=np.int32)
    py = df["py"].to_numpy(dtype=np.int32)
elif "pxl_col_in_fullres" in df.columns and "pxl_row_in_fullres" in df.columns:
    px = df["pxl_col_in_fullres"].to_numpy(dtype=np.int32)
    py = df["pxl_row_in_fullres"].to_numpy(dtype=np.int32)
else:
    raise KeyError("未能找到坐标列！请确认表格中包含 px/py 或 pxl_col_in_fullres/pxl_row_in_fullres。")

# 处理 cell_id，如果包含 NaN (未分配细胞的 bin) 则填充为 -1
if mask_key not in df.columns:
    raise KeyError(f"列 '{mask_key}' 不在 parquet 文件中！")

# 提取 cell_id 并且将缺失值(NaN)标记为 -1
cell_id = df[mask_key].fillna(-1).to_numpy()

# =========================================================
# 2. 读 H&E
# =========================================================
print("Loading H&E image...")
img = Image.open(img_path).convert("RGB")
W, H = img.size
print("Image size:", (W, H))

# =========================================================
# 3. 构建带 cell_id 的标签图
# =========================================================
label_img = np.zeros((H, W), dtype=np.int32)

# 过滤掉背景或无效细胞 (-1)
valid = (cell_id != 0)

x = px[valid]
y = py[valid]
cid = cell_id[valid].astype(np.int32)

# 防越界 (如果 Parquet 是全分辨率坐标，而图片是裁切好的，超出的点会被这里过滤掉)
in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
x = x[in_bounds]
y = y[in_bounds]
cid = cid[in_bounds]

# 每个像素写入对应 cell_id
label_img[y, x] = cid

print("Nonzero labeled pixels:", np.sum(label_img > 0))
print("Unique cells:", len(np.unique(cid)))

# =========================================================
# 4. 用固定大小圆形扩张，每个 bin 保留自己的 cell_id
# =========================================================
radius = 3   # 圆半径，可改成 3 / 4 / 5

label_img_expanded = np.zeros_like(label_img, dtype=np.int32)

# 预先生成一个圆形模板
yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
circle_mask = (xx**2 + yy**2) <= radius**2   # True 的位置构成圆

print("Expanding cell regions...")
for xi, yi, ci in zip(x, y, cid):
    y0 = max(0, yi - radius)
    y1 = min(H, yi + radius + 1)
    x0 = max(0, xi - radius)
    x1 = min(W, xi + radius + 1)

    # 对应裁剪后的圆形模板区域
    cy0 = y0 - (yi - radius)
    cy1 = cy0 + (y1 - y0)
    cx0 = x0 - (xi - radius)
    cx1 = cx0 + (x1 - x0)

    local_circle = circle_mask[cy0:cy1, cx0:cx1]

    # 只在圆内部赋值为该 cell_id
    sub = label_img_expanded[y0:y1, x0:x1]
    sub[local_circle] = ci
    label_img_expanded[y0:y1, x0:x1] = sub

print("Expanded labeled pixels:", np.sum(label_img_expanded > 0))

# =========================================================
# 5. 给不同 cell_id 分配不同颜色
# =========================================================
unique_ids = np.unique(label_img_expanded)
unique_ids = unique_ids[unique_ids > 0]
n_cells = len(unique_ids)
print("Expanded unique cells:", n_cells)

if n_cells > 0:
    # 建立 cell_id -> 连续编号 映射
    id_to_idx = {cid_: i for i, cid_ in enumerate(unique_ids)}

    # 连续编号图：0 为背景，1..n_cells 为不同细胞
    indexed_img = np.zeros_like(label_img_expanded, dtype=np.int32)
    mask_nonzero = (label_img_expanded > 0)
    indexed_img[mask_nonzero] = np.vectorize(id_to_idx.get)(label_img_expanded[mask_nonzero]) + 1

    # 生成较丰富的颜色表
    rng = np.random.default_rng(1234)
    hues = np.linspace(0, 1, n_cells, endpoint=False)
    rng.shuffle(hues)

    colors = np.zeros((n_cells + 1, 4), dtype=float)  # RGBA
    colors[0] = [0, 0, 0, 0]  # 背景透明

    for i, h in enumerate(hues, start=1):
        rgb = plt.cm.hsv(h)[:3]
        colors[i] = [rgb[0], rgb[1], rgb[2], 1.0]

    cell_cmap = ListedColormap(colors)
else:
    indexed_img = np.zeros_like(label_img_expanded, dtype=np.int32)
    cell_cmap = ListedColormap([[0, 0, 0, 0]])
    print("Warning: No valid cells found to colorize.")

# =========================================================
# 6. 可视化：不同细胞不同颜色
# =========================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.figure(figsize=(10, 10), dpi=200)
plt.imshow(img)
plt.axis("off")

overlay = np.ma.masked_where(indexed_img == 0, indexed_img)
plt.imshow(
    overlay,
    cmap=cell_cmap,
    alpha=0.50,
    interpolation="nearest"
)

plt.title("Cell segmentation mask overlay on H&E")
plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight", dpi=300)
print(f"PDF saved to {output_path}")

# =========================================================
# 7. 直接生成最终 RGB 图，便于后续切块保存
# =========================================================
print("Generating blended RGB image...")
base_img = np.array(img).astype(np.float32) / 255.0
final_img = base_img.copy()

alpha = 0.50

# 把每个像素对应的颜色取出来
rgba_img = cell_cmap(indexed_img)   # H x W x 4
rgb_img = rgba_img[..., :3]

mask_bool = (indexed_img > 0)
final_img[mask_bool] = (
    (1 - alpha) * final_img[mask_bool] +
    alpha * rgb_img[mask_bool]
)

final_img = np.clip(final_img * 255, 0, 255).astype(np.uint8)

# =========================================================
# 8. 按固定大小切块保存
# =========================================================
os.makedirs(out_dir, exist_ok=True)

tile_size = 512

H2, W2 = final_img.shape[:2]
print("Final image shape:", final_img.shape)

idx = 1
for top in range(0, H2, tile_size):
    bottom = min(top + tile_size, H2)

    for left in range(0, W2, tile_size):
        right = min(left + tile_size, W2)

        patch = final_img[top:bottom, left:right]

        save_path = os.path.join(out_dir, f"<OUTPUT_FIGURE_FILE>")
        Image.fromarray(patch).save(save_path)
        idx += 1

print(f"Done. Total patches saved: {idx - 1}")