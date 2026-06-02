import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# 0) R_HOME（已设可忽略）
os.environ["R_HOME"] = r"<R_HOME_PATH>"

r = ro.r

# 1) 输入输出路径（先统一成 R 可识别的路径格式）
#mat_csv = r"<CELLCHAT_OR_LR_OUTPUT_CSV>".replace("\\", "/")
#out_png = r"<OUTPUT_FIGURE_FILE>".replace("\\", "/")
#out_pdf = r"<OUTPUT_FIGURE_FILE>".replace("\\", "/")

mat_csv = r"<CELLCHAT_OR_LR_OUTPUT_CSV>".replace("\\", "/")
out_png = r"<SEGMENTATION_VISUALIZATION_FIGURE>".replace("\\", "/")
out_pdf = r"<SEGMENTATION_VISUALIZATION_FIGURE>".replace("\\", "/")

# 2) 画图（R 代码）
r(f"""
suppressPackageStartupMessages(library(CellChat))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(circlize))

mat <- read.csv("{mat_csv}", row.names=1, check.names=FALSE)
mat <- as.matrix(mat)

# 不画自环（可选）
diag(mat) <- 0

# log 缩放（非常推荐）
mat2 <- log1p(mat)

# PNG
png("{out_png}", width=1200, height=900, res=150)
netVisual_circle(
  mat2,
  weight.scale = TRUE,
  label.edge = FALSE,
  title.name = "CellChat-like overview",
  vertex.label.cex = 1.2
)
dev.off()

# PDF（论文用）
pdf("{out_pdf}", width=9, height=6)
netVisual_circle(
  mat2,
  weight.scale = TRUE,
  label.edge = FALSE,
  title.name = "CellChat-like overview",
  vertex.label.cex = 1.2
)
dev.off()
""")

print("Saved:")
print(out_png)
print(out_pdf)
