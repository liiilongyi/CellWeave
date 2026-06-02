import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import ListedColormap
import os
import tifffile

# ===== 关闭 PIL 大图限制 =====
Image.MAX_IMAGE_PIXELS = None

def read_ome_tiff(path):
    """
    专门用于读取 Xenium OME-TIFF 的函数
    优先读取最大的那个 Series (分辨率最高的层级)
    """
    print(f"   正在使用 tifffile 解析 OME-TIFF: {path}")
    try:
        with tifffile.TiffFile(path) as tif:
            # Xenium 的 OME-TIFF 通常 Series 0 是最高分辨率
            # 为了保险，我们遍历所有 series 找到像素数最多的那个
            largest_series = max(tif.series, key=lambda s: s.size)

            print(f"   检测到 {len(tif.series)} 个层级，选中形状: {largest_series.shape}")

            # 读取数据
            img = largest_series.asarray()
            return img
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        raise e


img_path = "<H_AND_E_IMAGE_FILE>"
#img_path = "<H_AND_E_IMAGE_FILE>"


seg_path = "<SEGMENTED_BIN_H5AD>"
#seg_path = "<SEGMENTED_BIN_H5AD>"

mask_key = "05_20_e290_9813"

#output_path = "<SEGMENTATION_VISUALIZATION_FIGURE>"
#out_dir = "<OUTPUT_DIRECTORY_OR_PREFIX>"

output_path = "<SEGMENTATION_VISUALIZATION_FIGURE>"
out_dir = "<OUTPUT_DIRECTORY_OR_PREFIX>"

# ===== 1. 读数据 =====
adata = sc.read_h5ad(seg_path)

px = adata.obs["px"].to_numpy(dtype=np.int32)
# px -= 19 * 2048   # colon part2 需要时再开

py = adata.obs["py"].to_numpy(dtype=np.int32)
cell_id = adata.obs[mask_key].to_numpy()

# ===== 2. 读 H&E =====
img = read_ome_tiff(img_path)
H, W = img.shape[:2]
print("Image size:", (W, H))

# =========================================================
# 3. 构建带 cell_id 的标签图，而不是二值图
# =========================================================
label_img = np.zeros((H, W), dtype=np.int32)

valid = (cell_id != -1)  # CellWeave --> 0;  others --> -1

x = px[valid]
y = py[valid]
cid = cell_id[valid].astype(np.int32)

# 防越界
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

# 建立 cell_id -> 连续编号 映射
id_to_idx = {cid_: i for i, cid_ in enumerate(unique_ids)}

# 连续编号图：0 为背景，1..n_cells 为不同细胞
indexed_img = np.zeros_like(label_img_expanded, dtype=np.int32)
mask_nonzero = (label_img_expanded > 0)
indexed_img[mask_nonzero] = np.vectorize(id_to_idx.get)(label_img_expanded[mask_nonzero]) + 1

# 生成较丰富的颜色表
# 用 HSV 均匀采样，避免全都接近
rng = np.random.default_rng(1234)
hues = np.linspace(0, 1, n_cells, endpoint=False)
rng.shuffle(hues)

colors = np.zeros((n_cells + 1, 4), dtype=float)  # RGBA
colors[0] = [0, 0, 0, 0]  # 背景透明

for i, h in enumerate(hues, start=1):
    rgb = plt.cm.hsv(h)[:3]
    colors[i] = [rgb[0], rgb[1], rgb[2], 1.0]

cell_cmap = ListedColormap(colors)

# =========================================================
# 6. 可视化：不同细胞不同颜色
# =========================================================
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
print("pdf saved")

# =========================================================
# 7. 直接生成最终 RGB 图，便于后续切块保存
# =========================================================
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