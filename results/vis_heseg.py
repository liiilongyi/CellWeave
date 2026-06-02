import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from matplotlib.colors import ListedColormap


# ===== 关闭 PIL 大图限制 =====
Image.MAX_IMAGE_PIXELS = None

img_path = "<H_AND_E_IMAGE_FILE>"
seg_path = "<INPUT_OR_OUTPUT_H5AD>"
mask_key = "<SEGMENTATION_MASK_COLUMN>"

#img_path = "<H_AND_E_IMAGE_FILE>"
#seg_path = "<INPUT_OR_OUTPUT_H5AD>"
#mask_key = "<SEGMENTATION_MASK_COLUMN>"

#img_path = "<LOCAL_DATA_OR_OUTPUT_PATH>"
#seg_path = "<INPUT_OR_OUTPUT_H5AD>"
#mask_key = "l3_cut13_ear_e1160_8945"

output_path = "<SEGMENTATION_VISUALIZATION_FIGURE>"
out_dir = "<OUTPUT_DIRECTORY_OR_PREFIX>"

# ===== 1. 读数据 =====
adata = sc.read_h5ad(seg_path)

px = adata.obs["px"].to_numpy(dtype=np.int32)       # px py 需减至图内
#px -= 19*2048                                      # colon part2 需变换

py = adata.obs["py"].to_numpy(dtype=np.int32)
cell_id = adata.obs[mask_key].to_numpy()

# ===== 2. 读 H&E =====
img = Image.open(img_path).convert("RGB")
W, H = img.size
print("Image size:", (W, H))

# ===== 3. 构建 binary mask（关键步骤）=====
# 用 uint8：0 或 1，省内存
mask_img = np.zeros((H, W), dtype=np.uint8)

# 只处理非零 bin
valid = (cell_id != 0)     # CellWeave --> 0;  others --> -1

x = px[valid]
y = py[valid]

# 防止极少数点越界
in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
x = x[in_bounds]
y = y[in_bounds]

# 标记为细胞区域
mask_img[y, x] = 1

# ===== 放大 mask（让小方格变大）=====
# 7x7 正方形结构元素
square7 = np.ones((8, 8), dtype=bool)

mask_img_dilated = binary_dilation(
    mask_img,
    structure=square7
).astype(np.uint8)


print("Mask filled pixels:", mask_img_dilated.sum())

overlay = np.ma.masked_where(mask_img_dilated == 0, mask_img_dilated)


# ===== 4. 可视化 =====
plt.figure(figsize=(10, 10), dpi=200)
# 底图
plt.imshow(img)
plt.axis("off")

# overlay：只给 mask=1 的地方上色
plt.imshow(
    overlay,
    #cmap="Reds",     # 单一色系
    cmap=ListedColormap(["#00D5FF"]),  # 你也可以试 "#00FFA8"（偏绿）
    alpha=0.58,       # 透明度     # 0.15  0.28
    interpolation="nearest"
)

plt.title("Cell segmentation mask overlay on H&E")
plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight", dpi=800)
plt.show()


import os

# ===== 5. 直接生成与可视化一致的 RGB 图 =====
base_img = np.array(img).astype(np.float32)   # (H, W, 3)

# overlay 颜色：与你 plt.imshow(... cmap=ListedColormap(["#00D5FF"])) 一致
overlay_color = np.array([0, 213, 255], dtype=np.float32)

alpha = 0.58

# 复制一份用于融合
final_img = base_img.copy()

# mask 区域
mask_bool = mask_img_dilated.astype(bool)

# alpha blending：final = (1-alpha)*base + alpha*color
final_img[mask_bool] = (
    (1 - alpha) * final_img[mask_bool] + alpha * overlay_color
)

final_img = np.clip(final_img, 0, 255).astype(np.uint8)

# ===== 6. 按固定大小切块保存 =====

os.makedirs(out_dir, exist_ok=True)

tile_size = 1024   # 你可以改成 512、2048 等

H2, W2 = final_img.shape[:2]
print("Final image shape:", final_img.shape)

idx = 1
for top in range(0, H2, tile_size):
    bottom = min(top + tile_size, H2)   # 最底部不足 tile_size 时，保留剩余高度

    for left in range(0, W2, tile_size):
        right = min(left + tile_size, W2)   # 最右边不足 tile_size 时，保留剩余宽度

        patch = final_img[top:bottom, left:right]

        save_path = os.path.join(out_dir, f"<OUTPUT_FIGURE_FILE>")
        Image.fromarray(patch).save(save_path)

        # print(f"saved: {save_path}, shape={patch.shape}")
        idx += 1

print(f"Done. Total patches saved: {idx - 1}")
