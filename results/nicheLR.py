import os
os.environ['R_HOME'] = r"<R_HOME_PATH>"

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from utils import clustering

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


# =========================================================
# 0. 参数区
# =========================================================
OUTDIR = "<OUTPUT_DIRECTORY_OR_PREFIX>"
os.makedirs(OUTDIR, exist_ok=True)

ADATA_PATH = "<CELL_LEVEL_EMBEDDING_H5AD>"
CELLCHAT_RDA = "<CELLCHAT_DB_RDA>"

RANDOM_STATE = 0
CAF_CLUSTER = "4"

# 你已有的 7-cluster 状态命名
state_map = {
    "0": "Metabolically Active/Stressed Tumor Cells",
    "1": "Ribosomal-High State",
    "2": "Oncofetal/developmental-like tumor state",
    "3": "Immunoreactive Tumor Cells",
    "4": "CAF / Stromal fibroblast",
    "5": "Rare/unstable",
    "6": "Tumor epithelial",
}

# CellChat-like 过滤阈值
pct_th_lig = 0.05
pct_th_rec = 0.05
mean_th_lig = 0.0
mean_th_rec = 0.0


# =========================================================
# 1. 通用函数
# =========================================================
def build_spatial_graph(xy, mode="knn", k=12, radius=None):
    n = xy.shape[0]

    if mode == "knn":
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(xy)
        dist, idx = nbrs.kneighbors(xy)
        src = np.repeat(np.arange(n), k)
        dst = idx[:, 1:].reshape(-1)

    elif mode == "radius":
        if radius is None:
            raise ValueError("radius mode requires radius")
        nbrs = NearestNeighbors(radius=radius, algorithm="auto").fit(xy)
        neigh_inds = nbrs.radius_neighbors(xy, return_distance=False)

        src_list, dst_list = [], []
        for i, arr in enumerate(neigh_inds):
            arr = arr[arr != i]
            if arr.size > 0:
                src_list.append(np.full(arr.shape[0], i, dtype=np.int64))
                dst_list.append(arr.astype(np.int64))

        if len(src_list) == 0:
            raise ValueError("No edges found under the chosen radius.")

        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)

    else:
        raise ValueError("mode must be 'knn' or 'radius'")

    keep = src < dst
    src_u = src[keep]
    dst_u = dst[keep]

    rows = np.concatenate([src_u, dst_u])
    cols = np.concatenate([dst_u, src_u])
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.eliminate_zeros()

    return src_u, dst_u, A


def choose_gmm_k(X, k_list, random_state=0, subsample=20000):
    rows = []
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    if n > subsample:
        sub_idx = rng.choice(n, size=subsample, replace=False)
        X_sub = X[sub_idx]
    else:
        X_sub = X

    for k in k_list:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            n_init=10
        )
        gmm.fit(X)
        labels = gmm.predict(X)

        sil = np.nan
        if len(np.unique(labels)) > 1:
            if X_sub.shape[0] == X.shape[0]:
                labels_sub = labels
            else:
                labels_sub = gmm.predict(X_sub)
            sil = silhouette_score(X_sub, labels_sub)

        rows.append({
            "k": k,
            "bic": gmm.bic(X),
            "aic": gmm.aic(X),
            "silhouette": sil
        })
    return pd.DataFrame(rows)


def spatial_scatter(adata, key, save=None, size=2, alpha=0.9, palette="tab20"):
    cats = adata.obs[key].astype("category")
    levels = cats.cat.categories.tolist()

    cmap = plt.get_cmap(palette, len(levels))
    color_map = {lv: cmap(i) for i, lv in enumerate(levels)}

    fig, ax = plt.subplots(figsize=(8, 8))
    for lv in levels:
        m = (cats == lv).values
        ax.scatter(
            adata.obs.loc[m, "px"],
            adata.obs.loc[m, "py"],
            s=size,
            c=[color_map[lv]],
            label=str(lv),
            alpha=alpha,
            linewidths=0
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=4)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()


def heatmap_plot(df, title, save, cmap="viridis", center=None, figsize=(6, 5), annot=False, fmt=".2f"):
    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap=cmap, center=center, annot=annot, fmt=fmt, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.close()


def geo_mean(vals):
    vals = np.asarray(vals, dtype=np.float32)
    vals = np.clip(vals, 1e-8, None)
    return float(np.exp(np.mean(np.log(vals))))


def parse_receptor_subunits(s: str):
    """
    从 CellChatDB.human$interaction 的 interaction_name_2 中解析受体亚基
    示例: "TGFB1 - (TGFBR1+TGFBR2)" -> ["TGFBR1", "TGFBR2"]
    """
    if not isinstance(s, str):
        return []
    m = re.search(r"\((.*?)\)", s)
    if not m:
        return []
    inner = m.group(1)
    parts = [p.strip() for p in inner.split("+") if p.strip()]
    return parts


def make_suggested_niche_names(niche_comp, state_map):
    rows = []
    for niche in niche_comp.index:
        s = niche_comp.loc[niche].sort_values(ascending=False)
        top1_cluster = str(s.index[0])
        top1_frac = float(s.iloc[0])

        if len(s) >= 2:
            top2_cluster = str(s.index[1])
            top2_frac = float(s.iloc[1])
        else:
            top2_cluster = top1_cluster
            top2_frac = 0.0

        top1_name = state_map.get(top1_cluster, f"Cluster {top1_cluster}")
        top2_name = state_map.get(top2_cluster, f"Cluster {top2_cluster}")

        suggested_name = f"Niche {niche}: {top1_name}"
        if top2_frac >= 0.20:
            suggested_name += f" + {top2_name}"

        rows.append({
            "niche_v2": str(niche),
            "top1_cluster": top1_cluster,
            "top1_cluster_name": top1_name,
            "top1_fraction": top1_frac,
            "top2_cluster": top2_cluster,
            "top2_cluster_name": top2_name,
            "top2_fraction": top2_frac,
            "suggested_niche_name": suggested_name
        })
    return pd.DataFrame(rows)


# =========================================================
# 2. 读数据 + 聚 7 类 cluster
# =========================================================
print("Loading adata ...")
adata = sc.read_h5ad(ADATA_PATH)

tool = "mclust"
clustering(
    adata,
    key="emb_latent_omics1",
    add_key="emb_latent_omics1",
    n_clusters=7,
    method=tool,
    use_pca=True
)
adata.obs["cluster"] = adata.obs["emb_latent_omics1"].astype("category")

# cluster 顺序与名字
adata.obs["cluster"] = adata.obs["cluster"].astype(str).astype("category")
cluster_order = sorted(
    adata.obs["cluster"].cat.categories.tolist(),
    key=lambda x: int(x) if str(x).isdigit() else str(x)
)
adata.obs["cluster"] = adata.obs["cluster"].cat.reorder_categories(cluster_order, ordered=True)
adata.obs["cluster_name"] = adata.obs["cluster"].astype(str).map(state_map).fillna(adata.obs["cluster"].astype(str))

print("Clustering done.")
print("clusters:", cluster_order)


# =========================================================
# 3. 读取 CellChatDB.human
# =========================================================
print("\nLoading CellChatDB.human ...")
r = ro.r
r(f'load("{CELLCHAT_RDA}")')

with localconverter(ro.default_converter + pandas2ri.converter):
    interaction_df = ro.conversion.rpy2py(r('CellChatDB.human$interaction'))

with localconverter(ro.default_converter + pandas2ri.converter):
    geneinfo_df = ro.conversion.rpy2py(r('CellChatDB.human$geneInfo'))

geneinfo_df.columns = [c.strip() for c in geneinfo_df.columns]

print("CellChat interaction rows:", interaction_df.shape[0])
print("CellChat geneInfo rows:", geneinfo_df.shape[0])


# =========================================================
# 4. 基因名匹配准备
# =========================================================
if "gene" not in adata.var.columns:
    raise KeyError("adata.var 中找不到 'gene' 列。请确认你的基因名存在哪一列。")

genes_raw = adata.var["gene"].astype(str).values
gene_list = genes_raw
gene_to_idx = {g: i for i, g in enumerate(gene_list)}

cellchat_symbols = set(geneinfo_df["Symbol"].dropna().astype(str))

# Ligand / receptor 子集
ligands = set(interaction_df["ligand"].dropna().astype(str))
receptor_subunits_all = set()

if "interaction_name_2" in interaction_df.columns:
    for s in interaction_df["interaction_name_2"].dropna().astype(str):
        receptor_subunits_all.update(parse_receptor_subunits(s))

adata_symbols = set(pd.Series(gene_list).dropna().astype(str))
lr_gene_set = set(ligands) | set(receptor_subunits_all)

print("\n================ CellChatDB 匹配率报告 ================")
print(f"adata genes/features: {len(gene_list)}")
print(f"CellChat 可识别 Symbol 命中数: {len(adata_symbols & cellchat_symbols)} / {len(adata_symbols)}")
print(f"ligand 命中数: {len(adata_symbols & ligands)} / {len(ligands)}")
print(f"receptor subunits 命中数: {len(adata_symbols & receptor_subunits_all)} / {len(receptor_subunits_all)}")
print(f"LR gene set 命中数: {len(adata_symbols & lr_gene_set)} / {len(lr_gene_set)}")
print("======================================================")


# =========================================================
# 5. 重建 niche_v2（沿用你原来的方式）
# =========================================================
print("\n[Build niche_v2] ...")

xy = np.c_[adata.obs["px"].values, adata.obs["py"].values].astype(float)
adata.obsm["spatial"] = xy

if "dist_to_caf_cd" not in adata.obs.columns:
    caf_mask = (adata.obs["cluster"].astype(str) == CAF_CLUSTER).values
    if caf_mask.sum() == 0:
        raise ValueError(f"CAF cluster {CAF_CLUSTER} not found.")

    tree = cKDTree(xy[caf_mask])
    dist_to_caf = tree.query(xy, k=1)[0]
    adata.obs["dist_to_caf"] = dist_to_caf

    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    d1 = nn.kneighbors(xy)[0][:, 1]
    base_scale = float(np.median(d1))
    adata.obs["dist_to_caf_cd"] = dist_to_caf / base_scale

src_u, dst_u, A = build_spatial_graph(xy, mode="knn", k=12, radius=None)

A_bin = A.copy().tocsr()
A_bin.data[:] = 1.0

deg = np.asarray(A_bin.sum(axis=1)).ravel()
deg_safe = np.maximum(deg, 1.0)
Dinv = sp.diags(1.0 / deg_safe)
A_norm = Dinv @ A_bin

cluster_onehot = pd.get_dummies(adata.obs["cluster"]).reindex(columns=cluster_order, fill_value=0)
cluster_onehot = cluster_onehot.to_numpy(dtype=float)

nbr1 = A_norm @ cluster_onehot
nbr2 = A_norm @ nbr1

caf_dist_feat = np.log1p(adata.obs["dist_to_caf_cd"].values).reshape(-1, 1)
X_niche = np.hstack([nbr1, nbr2, caf_dist_feat])

scaler = StandardScaler(with_mean=True, with_std=True)
X_niche_scaled = scaler.fit_transform(X_niche)

pca = PCA(n_components=min(12, X_niche_scaled.shape[1] - 1), random_state=RANDOM_STATE)
X_niche_pca = pca.fit_transform(X_niche_scaled)
adata.obsm["X_niche"] = X_niche_pca

k_df = choose_gmm_k(
    X_niche_pca,
    k_list=range(4, 7),
    random_state=RANDOM_STATE,
    subsample=20000
)
k_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)
print(k_df)

N_NICHES = int(k_df.sort_values(["silhouette", "bic"], ascending=[False, True]).iloc[0]["k"])
print("Selected N_NICHES (v2) =", N_NICHES)

gmm = GaussianMixture(
    n_components=N_NICHES,
    covariance_type="full",
    random_state=RANDOM_STATE,
    n_init=20
)
adata.obs["niche_v2"] = gmm.fit_predict(X_niche_pca).astype(str)
adata.obs["niche_v2"] = adata.obs["niche_v2"].astype("category")

niche_order = sorted(
    adata.obs["niche_v2"].cat.categories.tolist(),
    key=lambda x: int(x) if str(x).isdigit() else str(x)
)
adata.obs["niche_v2"] = adata.obs["niche_v2"].cat.reorder_categories(niche_order, ordered=True)

spatial_scatter(
    adata,
    "niche_v2",
    save=f"<SEGMENTATION_VISUALIZATION_FIGURE>",
    size=2
)

niche_comp_v2 = pd.crosstab(adata.obs["niche_v2"], adata.obs["cluster"], normalize="index")
niche_comp_v2.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>")

heatmap_plot(
    niche_comp_v2,
    title="Niche v2 composition by cluster",
    save=f"<SEGMENTATION_VISUALIZATION_FIGURE>",
    cmap="viridis",
    figsize=(8, 4),
    annot=True,
    fmt=".2f"
)

print("niche_v2 done.")
print("niches:", niche_order)


# =========================================================
# 6. 给每个 niche 一个建议名字
# =========================================================
niche_name_df = make_suggested_niche_names(niche_comp_v2, state_map)
niche_name_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)
print("\nSuggested niche names:")
print(niche_name_df[["niche_v2", "suggested_niche_name"]])


# =========================================================
# 7. 准备 CellChat-like LR 列表
#    这里直接用 CellChatDB.human$interaction
# =========================================================
pairs = []
for _, row in interaction_df.iterrows():
    lig = str(row["ligand"])

    rec_sub = []
    if "interaction_name_2" in interaction_df.columns:
        rec_sub = parse_receptor_subunits(row.get("interaction_name_2", ""))

    if len(rec_sub) == 0:
        # 若解析不到，直接跳过比硬用复合体名更稳
        continue

    pairs.append({
        "ligand": lig,
        "receptor_subunits": rec_sub,
        "pathway_name": row["pathway_name"] if "pathway_name" in interaction_df.columns else np.nan,
        "annotation": row["annotation"] if "annotation" in interaction_df.columns else np.nan,
        "interaction_name": row["interaction_name"] if "interaction_name" in interaction_df.columns else np.nan,
        "interaction_name_2": row["interaction_name_2"] if "interaction_name_2" in interaction_df.columns else np.nan,
    })

pairs_df = pd.DataFrame(pairs).drop_duplicates(subset=["ligand", "interaction_name_2"]).reset_index(drop=True)
pairs_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)
print("\nParsed usable LR pairs:", pairs_df.shape[0])


# =========================================================
# 8. 改成以 niche_v2 为 group，做 4x4 niche 通讯矩阵
# =========================================================
group_key = "niche_v2"
celltype = adata.obs[group_key].astype(str).values
types = pd.Categorical(celltype).categories.tolist()
n_types = len(types)
type_to_cells = {t: np.where(celltype == t)[0] for t in types}

X = adata.X
if not sp.issparse(X):
    X = sp.csr_matrix(X)

mean_expr = np.zeros((n_types, X.shape[1]), dtype=np.float32)
pct_expr = np.zeros((n_types, X.shape[1]), dtype=np.float32)

print("\nPrecomputing mean / pct expression per niche ...")
for ti, t in enumerate(types):
    idx = type_to_cells[t]
    Xsub = X[idx, :]
    mean_expr[ti, :] = np.asarray(Xsub.mean(axis=0)).ravel()
    nnz = np.diff(Xsub.tocsc().indptr)
    pct_expr[ti, :] = nnz / max(len(idx), 1)

print("done.")


# =========================================================
# 9. 计算 niche->niche 通讯矩阵
# =========================================================
print("\nComputing niche-to-niche CellChat-like matrix ...")

W = np.zeros((n_types, n_types), dtype=np.float64)
edge_rows = []

for _, row in pairs_df.iterrows():
    lig = row["ligand"]
    rec_list = row["receptor_subunits"]

    if lig not in gene_to_idx:
        continue
    lig_j = gene_to_idx[lig]

    rec_idx = [gene_to_idx[r] for r in rec_list if r in gene_to_idx]
    if len(rec_idx) != len(rec_list):
        # 复合体亚基没凑齐，就跳过
        continue
    if len(rec_idx) == 0:
        continue

    lig_mean_all = mean_expr[:, lig_j]
    lig_pct_all = pct_expr[:, lig_j]

    rec_mean_all = mean_expr[:, rec_idx]   # [n_types, k]
    rec_pct_all = pct_expr[:, rec_idx]

    for a in range(n_types):   # sender niche
        if lig_pct_all[a] < pct_th_lig or lig_mean_all[a] < mean_th_lig:
            continue

        lig_val = float(lig_mean_all[a])

        for b in range(n_types):  # receiver niche
            if np.any(rec_pct_all[b] < pct_th_rec) or np.any(rec_mean_all[b] < mean_th_rec):
                continue

            rec_val = geo_mean(rec_mean_all[b])
            score = lig_val * rec_val

            W[a, b] += score

            edge_rows.append({
                "sender_niche": types[a],
                "receiver_niche": types[b],
                "ligand": lig,
                "receptor_subunits": ";".join(rec_list),
                "score": score,
                "pathway_name": row["pathway_name"],
                "annotation": row["annotation"],
                "interaction_name": row["interaction_name"],
                "interaction_name_2": row["interaction_name_2"],
            })

# 保存详细 pair-level 结果
edge_df = pd.DataFrame(edge_rows)
edge_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)


# =========================================================
# 10. 保存 4x4 矩阵
# =========================================================
W_df = pd.DataFrame(W, index=types, columns=types)
W_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>")

W_log = np.log1p(W)
W_log_df = pd.DataFrame(W_log, index=types, columns=types)
W_log_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>")

W_norm = (W_log - W_log.min()) / (W_log.max() - W_log.min() + 1e-12)
W_norm_df = pd.DataFrame(W_norm, index=types, columns=types)
W_norm_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>")

print("Saved:",
      f"<CELLCHAT_OR_LR_OUTPUT_CSV>",
      f"<CELLCHAT_OR_LR_OUTPUT_CSV>",
      f"<CELLCHAT_OR_LR_OUTPUT_CSV>")


# =========================================================
# 11. 画热图
# =========================================================
heatmap_plot(
    W_df,
    title="Niche-to-niche CellChat-like strength",
    save=f"<SEGMENTATION_VISUALIZATION_FIGURE>",
    cmap="Reds",
    figsize=(6, 5),
    annot=True,
    fmt=".2f"
)

heatmap_plot(
    W_log_df,
    title="Niche-to-niche CellChat-like strength (log1p)",
    save=f"<SEGMENTATION_VISUALIZATION_FIGURE>",
    cmap="Reds",
    figsize=(6, 5),
    annot=True,
    fmt=".2f"
)

heatmap_plot(
    W_norm_df,
    title="Niche-to-niche CellChat-like strength (normalized)",
    save=f"<SEGMENTATION_VISUALIZATION_FIGURE>",
    cmap="Reds",
    figsize=(6, 5),
    annot=True,
    fmt=".2f"
)


# =========================================================
# 12. 顺手保存每个 niche 最强的 outgoing / incoming
# =========================================================
summary_rows = []
for i, sender in enumerate(types):
    for j, receiver in enumerate(types):
        summary_rows.append({
            "sender_niche": sender,
            "receiver_niche": receiver,
            "score": W[i, j]
        })
summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
summary_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)

top_out_rows = []
top_in_rows = []

for i, sender in enumerate(types):
    row = W[i, :]
    j = int(np.argmax(row))
    top_out_rows.append({
        "niche": sender,
        "top_outgoing_to": types[j],
        "top_outgoing_score": row[j]
    })

for j, receiver in enumerate(types):
    col = W[:, j]
    i = int(np.argmax(col))
    top_in_rows.append({
        "niche": receiver,
        "top_incoming_from": types[i],
        "top_incoming_score": col[i]
    })

pd.DataFrame(top_out_rows).to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)
pd.DataFrame(top_in_rows).to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)

print("\nDone.")
print("Main outputs:")
print(f"<CELLCHAT_OR_LR_OUTPUT_CSV>")
print(f"<CELLCHAT_OR_LR_OUTPUT_CSV>")
print(f"<CELLCHAT_OR_LR_OUTPUT_CSV>")






# =========================================================
# 13. 第二层：pathway-level decomposition
# =========================================================
print("\n[13] Pathway-level decomposition ...")

pairwise_path = f"<CELLCHAT_OR_LR_OUTPUT_CSV>"
edge_df = pd.read_csv(pairwise_path)

# 去掉没有 pathway 的条目
edge_df["pathway_name"] = edge_df["pathway_name"].astype(str)
edge_df = edge_df[edge_df["pathway_name"].notna()].copy()
edge_df = edge_df[edge_df["pathway_name"] != "nan"].copy()
edge_df = edge_df[edge_df["pathway_name"] != ""].copy()

# 1) pathway 总表：每个 pathway 在每条 edge 上的总分
pathway_edge_df = (
    edge_df.groupby(["sender_niche", "receiver_niche", "pathway_name"], as_index=False)
    .agg(
        pathway_score=("score", "sum"),
        n_pairs=("score", "size"),
        mean_pair_score=("score", "mean")
    )
    .sort_values("pathway_score", ascending=False)
)

pathway_edge_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)

# 2) 全局 top pathways
pathway_overall_df = (
    pathway_edge_df.groupby("pathway_name", as_index=False)
    .agg(
        total_score=("pathway_score", "sum"),
        total_pairs=("n_pairs", "sum")
    )
    .sort_values("total_score", ascending=False)
)

pathway_overall_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)

top_pathways = pathway_overall_df.head(12)["pathway_name"].tolist()

plt.figure(figsize=(7, 4))
sns.barplot(
    data=pathway_overall_df.head(12),
    x="total_score",
    y="pathway_name"
)
plt.title("Top pathways overall")
plt.tight_layout()
plt.savefig(f"<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
plt.close()

# 3) 看 strongest edge: 1 -> 3
edge_13_df = pathway_edge_df[
    (pathway_edge_df["sender_niche"].astype(str) == "1") &
    (pathway_edge_df["receiver_niche"].astype(str) == "3")
].sort_values("pathway_score", ascending=False)

edge_13_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)

plt.figure(figsize=(7, 4))
sns.barplot(
    data=edge_13_df.head(12),
    x="pathway_score",
    y="pathway_name"
)
plt.title("Top pathways for niche 1 -> niche 3")
plt.tight_layout()
plt.savefig(f"<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
plt.close()

# 4) 看第二强边: 3 -> 3
edge_33_df = pathway_edge_df[
    (pathway_edge_df["sender_niche"].astype(str) == "3") &
    (pathway_edge_df["receiver_niche"].astype(str) == "3")
].sort_values("pathway_score", ascending=False)

edge_33_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)

plt.figure(figsize=(7, 4))
sns.barplot(
    data=edge_33_df.head(12),
    x="pathway_score",
    y="pathway_name"
)
plt.title("Top pathways for niche 3 -> niche 3")
plt.tight_layout()
plt.savefig(f"<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
plt.close()

# 5) 选 top 6 pathways，画 bubble plot
selected_pathways = pathway_overall_df.head(6)["pathway_name"].tolist()
plot_df = pathway_edge_df[pathway_edge_df["pathway_name"].isin(selected_pathways)].copy()
plot_df["edge"] = plot_df["sender_niche"].astype(str) + "->" + plot_df["receiver_niche"].astype(str)

plt.figure(figsize=(8, 4))
sns.scatterplot(
    data=plot_df,
    x="edge",
    y="pathway_name",
    size="pathway_score",
    hue="pathway_score",
    sizes=(20, 400),
    palette="Reds"
)
plt.title("Selected pathways across niche-to-niche edges")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
plt.close()

# 6) 每个 pathway 单独做 4x4 热图矩阵
for pw in selected_pathways:
    tmp = pathway_edge_df[pathway_edge_df["pathway_name"] == pw].copy()
    mat = tmp.pivot_table(
        index="sender_niche",
        columns="receiver_niche",
        values="pathway_score",
        aggfunc="sum",
        fill_value=0
    )
    # 保证 0/1/2/3 顺序
    mat = mat.reindex(index=types, columns=types, fill_value=0)
    mat.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>")

    plt.figure(figsize=(5, 4))
    sns.heatmap(mat, cmap="Reds", annot=True, fmt=".2f", square=True)
    plt.title(f"{pw} pathway across niches")
    plt.tight_layout()
    plt.savefig(f"<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
    plt.close()

print("Pathway-level decomposition done.")

# =========================================================
# 14. 第三层：top LR pairs for key edges
# =========================================================
print("\n[14] Top LR pairs for key edges ...")

pairwise_path = f"<CELLCHAT_OR_LR_OUTPUT_CSV>"
edge_df = pd.read_csv(pairwise_path)

# 定义一个 pair ID
edge_df["lr_pair"] = edge_df["ligand"].astype(str) + " -> " + edge_df["receptor_subunits"].astype(str)

# 先做每个 edge 的 pair score 表
lr_edge_df = (
    edge_df.groupby(["sender_niche", "receiver_niche", "lr_pair", "ligand", "receptor_subunits", "pathway_name"], as_index=False)
    .agg(pair_score=("score", "sum"))
)

lr_edge_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)

# -----------------------------
# A. 看 strongest edge 1 -> 3
# -----------------------------
edge_key = ("1", "3")
edge_target_df = lr_edge_df[
    (lr_edge_df["sender_niche"].astype(str) == edge_key[0]) &
    (lr_edge_df["receiver_niche"].astype(str) == edge_key[1])
].copy()

# 计算 edge specificity
other_df = lr_edge_df[
    ~(
        (lr_edge_df["sender_niche"].astype(str) == edge_key[0]) &
        (lr_edge_df["receiver_niche"].astype(str) == edge_key[1])
    )
].copy()

other_med = (
    other_df.groupby("lr_pair", as_index=False)
    .agg(other_edges_median=("pair_score", "median"))
)

edge_target_df = edge_target_df.merge(other_med, on="lr_pair", how="left")
edge_target_df["other_edges_median"] = edge_target_df["other_edges_median"].fillna(0.0)
edge_target_df["specificity_ratio"] = edge_target_df["pair_score"] / (edge_target_df["other_edges_median"] + 1e-8)

# 综合排序：先看强度，再看特异性
edge_target_df = edge_target_df.sort_values(
    ["pair_score", "specificity_ratio"],
    ascending=[False, False]
)

edge_target_df.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)
edge_target_df.head(20).to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=edge_target_df.head(15),
    x="pair_score",
    y="lr_pair"
)
plt.title("Top LR pairs for niche 1 -> niche 3")
plt.tight_layout()
plt.savefig(f"<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# B. 再看 3 -> 3
# -----------------------------
edge_key = ("3", "3")
edge_target_df_33 = lr_edge_df[
    (lr_edge_df["sender_niche"].astype(str) == edge_key[0]) &
    (lr_edge_df["receiver_niche"].astype(str) == edge_key[1])
].copy()

other_df_33 = lr_edge_df[
    ~(
        (lr_edge_df["sender_niche"].astype(str) == edge_key[0]) &
        (lr_edge_df["receiver_niche"].astype(str) == edge_key[1])
    )
].copy()

other_med_33 = (
    other_df_33.groupby("lr_pair", as_index=False)
    .agg(other_edges_median=("pair_score", "median"))
)

edge_target_df_33 = edge_target_df_33.merge(other_med_33, on="lr_pair", how="left")
edge_target_df_33["other_edges_median"] = edge_target_df_33["other_edges_median"].fillna(0.0)
edge_target_df_33["specificity_ratio"] = edge_target_df_33["pair_score"] / (edge_target_df_33["other_edges_median"] + 1e-8)

edge_target_df_33 = edge_target_df_33.sort_values(
    ["pair_score", "specificity_ratio"],
    ascending=[False, False]
)

edge_target_df_33.to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)
edge_target_df_33.head(20).to_csv(f"<CELLCHAT_OR_LR_OUTPUT_CSV>", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=edge_target_df_33.head(15),
    x="pair_score",
    y="lr_pair"
)
plt.title("Top LR pairs for niche 3 -> niche 3")
plt.tight_layout()
plt.savefig(f"<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# C. 做一个 edge × LR 的 dotplot
# -----------------------------
top_pairs = pd.concat([
    edge_target_df.head(10)[["lr_pair"]],
    edge_target_df_33.head(10)[["lr_pair"]]
], axis=0).drop_duplicates()["lr_pair"].tolist()

show_df = lr_edge_df[lr_edge_df["lr_pair"].isin(top_pairs)].copy()
show_df["edge"] = show_df["sender_niche"].astype(str) + "->" + show_df["receiver_niche"].astype(str)

plt.figure(figsize=(8, max(4, len(top_pairs) * 0.3)))
sns.scatterplot(
    data=show_df,
    x="edge",
    y="lr_pair",
    size="pair_score",
    hue="pair_score",
    sizes=(20, 350),
    palette="Reds"
)
plt.title("Top LR pairs across niche-to-niche edges")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
plt.close()

print("Top LR pair analysis done.")