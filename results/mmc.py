import pandas as pd
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
#adata = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")


mmc = pd.read_csv("<MAPMYCELLS_ANNOTATION_CSV>", comment="#")  # 自动跳过开头的#注释行
#mmc = pd.read_csv("<MAPMYCELLS_ANNOTATION_CSV>", comment="#")  # 自动跳过开头的#注释行

mmc["cell_id"] = mmc["cell_id"].astype(str)
mmc = mmc.set_index("cell_id")

cols = [
    "class_name", "subclass_name", "supertype_name", "cluster_name",
    "class_bootstrapping_probability", "subclass_bootstrapping_probability",
    "supertype_bootstrapping_probability", "cluster_bootstrapping_probability"
]

for c in cols:
    adata.obs["mmc_" + c] = mmc.reindex(adata.obs_names)[c].values

label = "mmc_class_name"
cats = adata.obs[label].astype("category")

codes = cats.cat.codes.to_numpy()
names = cats.cat.categories.tolist()
n_cat = len(names)

# ① 选择区分度大的离散 colormap
# class 数量一般 < 20，用 tab20 最合适
cmap = plt.get_cmap("tab20", n_cat)

colors = cmap(codes)

plt.figure(figsize=(18, 18))
plt.scatter(adata.obs["px"].values, adata.obs["py"].values, c=colors, s=8, linewidths=0)

plt.gca().invert_yaxis()
plt.axis("equal")
plt.title(label)

# ② 构造右侧 legend（颜色 ↔ 类型名）
handles = [
    Line2D(
        [0], [0],
        marker='o',
        linestyle='',
        markerfacecolor=cmap(i),
        markeredgecolor='none',
        markersize=6,
        label=names[i]
    )
    for i in range(n_cat)
]

plt.legend(handles=handles, title="Cell type", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

plt.tight_layout()

out_png = "<OUTPUT_FIGURE_FILE>"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
