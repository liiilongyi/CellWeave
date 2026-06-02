import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import scipy.sparse as sp
import matplotlib.pyplot as plt
from utils import clustering


def main():
    adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 用来聚类（embedding）

    tool = "mclust"
    clustering(adata, key="emb_latent_omics1", add_key="emb_latent_omics1", n_clusters=13, method=tool, use_pca=True)
    adata.obs["celltype"] = adata.obs["emb_latent_omics1"].astype("category")

    adata.obsm["spatial"] = np.c_[adata.obs["px"].values, adata.obs["py"].values].astype(float)

    # =========================================================
    # 1) 构建空间邻接图：kNN（建议先用kNN，后面再做k敏感性分析）
    # =========================================================
    K = 15  # 建议 10/15/20 做稳定性检查
    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        n_neighs=K,  # ← 你的版本正确参数名
        delaunay=False,
        set_diag=False,  # 可选：一般不要自环
        key_added="spatial"
    )

    # （可选）看一下邻接图是否正常：每个细胞平均邻居数
    A = adata.obsp["spatial_connectivities"].tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    print("mean degree:", deg.mean(), "min:", deg.min(), "max:", deg.max())

    # =========================================================
    # 2) 邻域富集检验（最“科学”的那张图）：nhood enrichment heatmap
    # =========================================================
    sq.gr.nhood_enrichment(adata, cluster_key="celltype")
    sq.pl.nhood_enrichment(
        adata,
        cluster_key="celltype",
        figsize=(10, 10)
    )

    # 如果类别太多（比如 cluster_name 上百类），建议先用 class/subclass 或者挑TopN类
    # 例如：保留细胞数最多的前30类再做富集图（可选）：
    # topN = 30
    # top_types = adata.obs["celltype"].value_counts().head(topN).index
    # adata_top = adata[adata.obs["celltype"].isin(top_types)].copy()
    # sq.gr.spatial_neighbors(adata_top, coord_type="generic", n_neigh=K, delaunay=False)
    # sq.gr.nhood_enrichment(adata_top, cluster_key="celltype")
    # sq.pl.nhood_enrichment(adata_top, cluster_key="celltype", figsize=(7,6))

    # =========================================================
    # 3) 每个细胞的“邻居是谁”：计算邻域组成向量（比例/计数）
    # =========================================================
    def neighbor_composition(
            adata,
            celltype_key="celltype",
            conn_key="spatial_connectivities",
            out_key="nhood_comp",
    ):
        A = adata.obsp[conn_key].tocsr()
        ct = adata.obs[celltype_key].astype("category")
        cats = ct.cat.categories

        # one-hot: [n_cells, n_types]
        onehot = sp.csr_matrix(
            (np.ones(adata.n_obs, dtype=np.float32),
             (np.arange(adata.n_obs), ct.cat.codes)),
            shape=(adata.n_obs, len(cats))
        )

        neigh_counts = A @ onehot  # [n_cells, n_types]
        neigh_counts = neigh_counts.toarray()

        row_sum = neigh_counts.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        neigh_prop = neigh_counts / row_sum

        adata.obsm[out_key] = neigh_prop
        adata.obsm[out_key + "_counts"] = neigh_counts
        adata.uns[out_key + "_cats"] = list(cats)

    neighbor_composition(adata, celltype_key="celltype")

    cats = adata.uns["nhood_comp_cats"]
    P = pd.DataFrame(adata.obsm["nhood_comp"], columns=cats, index=adata.obs_names)

    # =========================================================
    # 4) 汇总可视化：按“中心细胞类型”平均邻域组成（堆叠条形图）
    #    （非常直观：每类细胞周围都有哪些邻居）
    # =========================================================
    avgP = P.groupby(adata.obs["celltype"]).mean()

    # 如果细胞类型太多，建议只画TopN中心类型
    TOP_CENTER = 25
    top_center_types = adata.obs["celltype"].value_counts().head(TOP_CENTER).index
    avgP_top = avgP.loc[top_center_types]



    # 图例放到外面（更美观）
    fig, ax = plt.subplots(figsize=(15, 5))
    avgP_top.plot(kind="bar", stacked=True, ax=ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=1, title="Neighbor type")
    ax.set_ylabel("Mean neighbor proportion")
    ax.set_xlabel("Focal cell type")
    ax.set_title(f"Local neighborhood composition (k={K})")
    plt.tight_layout()
    plt.show()

    # =========================================================
    # 5) 先进玩法：用邻域组成向量做“niche（空间微环境）”聚类
    #    （这一步通常最“好看”、最像CellChat那类高级总结图）
    # =========================================================
    adata.obsm["X_nhood"] = adata.obsm["nhood_comp"].astype(np.float32)

    sc.pp.neighbors(adata, use_rep="X_nhood", n_neighbors=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.6, key_added="niche")

    # UMAP 看 niche 是否分离
    sc.pl.umap(adata, color=["niche", "celltype"], wspace=0.4)

    # niche 映射回空间：最关键的一张
    sc.pl.embedding(
        adata,
        basis="spatial",
        color=["niche", "celltype"],
        size=6,
        frameon=False,
        ncols=2
    )

    # =========================================================
    # 6) 解释 niche：每个 niche 的平均邻域组成（热图）
    # =========================================================
    niche_avg = P.groupby(adata.obs["niche"]).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(niche_avg.values, aspect="auto")
    ax.set_yticks(range(niche_avg.shape[0]))
    ax.set_yticklabels(niche_avg.index)
    ax.set_xticks(range(niche_avg.shape[1]))
    ax.set_xticklabels(niche_avg.columns, rotation=90)
    ax.set_title("Mean neighborhood composition per niche")
    plt.colorbar(im, ax=ax, label="Mean neighbor proportion")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()