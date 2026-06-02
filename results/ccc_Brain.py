import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import os

os.environ['R_HOME'] = "<R_HOME_PATH>"
csv_path = "<OUTPUT_CSV_FILE>"
out_pdf  = "<OUTPUT_FIGURE_FILE>"
out_png  = "<OUTPUT_FIGURE_FILE>"

adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
mmc = pd.read_csv("<MAPMYCELLS_ANNOTATION_CSV>", comment="#")  # 自动跳过开头的#注释行

mmc["cell_id"] = mmc["cell_id"].astype(str)
mmc = mmc.set_index("cell_id")

cols = [
    "class_name", "subclass_name", "supertype_name", "cluster_name",
    "class_bootstrapping_probability", "subclass_bootstrapping_probability",
    "supertype_bootstrapping_probability", "cluster_bootstrapping_probability"
]

for c in cols:
    adata.obs["mmc_" + c] = mmc.reindex(adata.obs_names)[c].values

# 确保坐标
xy = np.c_[adata.obs["px"].values, adata.obs["py"].values].astype(float)

# cluster
clusters = adata.obs["mmc_class_name"].astype("category")
names = clusters.cat.categories.tolist()
codes = clusters.cat.codes.to_numpy()
K = len(names)

# --- 建 kNN 图（你现在用 8，也可以试 12/16 看跨群连接会不会更多）
KNN = 12
nbrs = NearestNeighbors(n_neighbors=KNN+1, algorithm="auto").fit(xy)
dist, idx = nbrs.kneighbors(xy)

# idx[:,0] 是自己，去掉
src = np.repeat(np.arange(xy.shape[0]), KNN)
dst = idx[:, 1:].reshape(-1)

# 去重：只保留 i<j 的无向边
m = src < dst
src_u = src[m]
dst_u = dst[m]

t1 = codes[src_u]
t2 = codes[dst_u]

# cluster-cluster 边计数矩阵
mat = np.zeros((K, K), dtype=np.int64)
for a, b in zip(t1, t2):
    if a <= b:
        mat[a, b] += 1
    else:
        mat[b, a] += 1
mat = mat + mat.T
np.fill_diagonal(mat, np.diag(mat)//2)  # 对角保持原本计数语义（同类边）

df = pd.DataFrame(mat, index=names, columns=names)
df.to_csv(csv_path)
print("<OUTPUT_CSV_FILE>")





circlize = importr("circlize")
grdevices = importr("grDevices")

# R 脚本：读矩阵 -> 过滤小边 -> 画 chord
r_code = f"""
library(circlize)

mat <- as.matrix(read.csv("{csv_path}", row.names=1, check.names=FALSE))

# --- 可选：过滤很小的连接，让弦更“干净”
# 你可以调这个分位数：0.5~0.8
thr <- quantile(mat[mat>0], 0.60)
mat2 <- mat
mat2[mat2 < thr] <- 0

# 给每个 cluster 一个明显颜色
grid.col <- setNames(rainbow(nrow(mat2)), rownames(mat2))

# 输出 PDF
pdf("{out_pdf}", width=10, height=10)
circos.clear()
chordDiagram(
  mat2,
  grid.col = grid.col,
  transparency = 0.7,
  directional = 0,
  annotationTrack = "grid",
  preAllocateTracks = 1
)

# 外圈标签
circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {{
  sector.name = get.cell.meta.data("sector.index")
  xlim = get.cell.meta.data("xlim")
  ylim = get.cell.meta.data("ylim")
  circos.text(mean(xlim), ylim[1] + .1, sector.name,
              facing="clockwise", niceFacing=TRUE, adj=c(0, 0.5), cex=0.7)
}}, bg.border = NA)
dev.off()

# 同时输出 PNG（方便你快速看）
png("{out_png}", width=3000, height=3000, res=300)
circos.clear()
chordDiagram(
  mat2,
  grid.col = grid.col,
  transparency = 0.7,
  directional = 0,
  annotationTrack = "grid",
  preAllocateTracks = 1
)
circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {{
  sector.name = get.cell.meta.data("sector.index")
  xlim = get.cell.meta.data("xlim")
  ylim = get.cell.meta.data("ylim")
  circos.text(mean(xlim), ylim[1] + .1, sector.name,
              facing="clockwise", niceFacing=TRUE, adj=c(0, 0.5), cex=0.7)
}}, bg.border = NA)
dev.off()
"""

robjects.r(r_code)
print("[Saved]", out_pdf)
print("[Saved]", out_png)