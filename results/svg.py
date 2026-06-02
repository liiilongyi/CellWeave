import numpy as np
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

def main():
    adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")
    #adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")

    # 坐标
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = np.c_[adata.obs["px"].values, adata.obs["py"].values].astype(float)
    else:
        adata.obsm["spatial"] = np.c_[adata.obs["px"].values, adata.obs["py"].values].astype(float)

    adata.var_names = adata.var["gene"].astype(str)
    adata.var_names_make_unique()
    adata.var.index.name = None

    # 邻接
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=8)

    # SVG（先别置换，保证跑通）
    sq.gr.spatial_autocorr(adata, mode="moran", n_perms=None)

    moran_df = adata.uns["moranI"].sort_values("I", ascending=False)
    print(moran_df.head(10))
    #moran_df.to_csv("<OUTPUT_CSV_FILE>")

    # 画前 9 个
    top_genes = moran_df.index[:9].tolist()
    out_pdf = "<OUTPUT_FIGURE_FILE>"

    sc.pl.embedding(
        adata,
        basis="spatial",
        color=top_genes,
        s=2,
        ncols=3,
        show=False,  # 关键：不直接弹窗
        save=None  # 先别用scanpy自动save，避免路径限制
    )

    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.show()
    #plt.close()

if __name__ == "__main__":
    # Windows multiprocessing 必须有
    main()