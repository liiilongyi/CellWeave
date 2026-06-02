import os
os.environ['R_HOME'] = r"<R_HOME_PATH>"

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import numpy as np
import scanpy as sc
from utils import clustering
import re
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Arc
import matplotlib.cm as cm
from collections import defaultdict

adata = sc.read_h5ad("<CELL_LEVEL_EMBEDDING_H5AD>")  # 用来聚类（embedding）

tool = "mclust"
clustering(adata, key="emb_latent_omics1", add_key="emb_latent_omics1", n_clusters=13, method=tool, use_pca=True)
adata.obs["cluster"] = adata.obs["emb_latent_omics1"].astype("category")
print("go")

r = ro.r
# 加载 .rda 文件（注意是 R 的 load）
r('<OUTPUT_DIRECTORY_OR_PREFIX>')

# 看当前 R 环境里有什么对象
#print(r('ls()'))
# 查看对象结构（等价于 R 里的 str()）
#print(r('str(CellChatDB.human, max.level=2)'))


with localconverter(ro.default_converter + pandas2ri.converter):
    interaction_df = ro.conversion.rpy2py(r('CellChatDB.human$interaction'))

with localconverter(ro.default_converter + pandas2ri.converter):
    geneinfo_df = ro.conversion.rpy2py(r('CellChatDB.human$geneInfo'))

# 规范列名（防止大小写/空格差异）
geneinfo_df.columns = [c.strip() for c in geneinfo_df.columns]

# 建立 Ensembl -> Symbol 映射
# 注意：geneInfo 里 Ensembl.Gene.ID 可能为空/NA，要过滤
ens2sym = (
    geneinfo_df[["Ensembl.Gene.ID", "Symbol"]]
    .dropna()
    .drop_duplicates()
)
ens2sym = dict(zip(ens2sym["Ensembl.Gene.ID"].astype(str), ens2sym["Symbol"].astype(str)))

# CellChatDB 里所有“基因符号”集合（用于匹配率计算）
cellchat_symbols = set(geneinfo_df["Symbol"].dropna().astype(str))

# =========================================================
# 1) 取出你的基因列表（来自 adata.var["gene"]）
# =========================================================
if "gene" not in adata.var.columns:
    raise KeyError("adata.var 中找不到 'gene' 列。请确认你的基因名存在哪一列。")

gene_raw = adata.var["gene"].astype(str).values  # 长度 = n_vars

# 判断 Ensembl ID 的正则（ENSGxxxx）
ens_pat = re.compile(r"^ENSG\d+")

is_ens = np.array([bool(ens_pat.match(g)) for g in gene_raw])

# 映射：Ensembl -> Symbol；Symbol 原样保留
gene_symbol = np.array(gene_raw, dtype=object)
# 仅对 Ensembl 的条目做映射（映射不到就保留原 ENSG）
gene_symbol[is_ens] = [ens2sym.get(g, g) for g in gene_raw[is_ens]]

# 统计映射成功数：Ensembl 且映射后不是 ENSG 开头
mapped_success = is_ens & np.array([not bool(ens_pat.match(gs)) for gs in gene_symbol])

# =========================================================
# 2) 计算 CellChatDB 的 ligand / receptor（注意 receptor 可能是复合体名）
#    - ligand 列基本都是 Symbol，可直接匹配
#    - receptor 列有很多是复合体标签（如 TGFbR1_R2），不能直接当基因
#      所以：这里给两种 receptor 匹配率：
#         (a) 直接把 receptor 当成 symbol 匹配（会低，偏保守）
#         (b) 从 interaction_name_2 解析括号内的真实亚基基因（更接近真实）
# =========================================================
ligands = set(interaction_df["ligand"].dropna().astype(str))

# (a) receptor 直接匹配（保守）
receptors_raw = set(interaction_df["receptor"].dropna().astype(str))

# (b) 从 interaction_name_2 解析括号内容，如 "TGFB1 - (TGFBR1+TGFBR2)"
def parse_receptor_subunits(s: str):
    if not isinstance(s, str):
        return []
    m = re.search(r"\((.*?)\)", s)
    if not m:
        return []
    inner = m.group(1)
    # split by '+' and remove spaces
    parts = [p.strip() for p in inner.split("+") if p.strip()]
    return parts

receptor_subunits = set()
if "interaction_name_2" in interaction_df.columns:
    for s in interaction_df["interaction_name_2"].dropna().astype(str):
        receptor_subunits.update(parse_receptor_subunits(s))

# 你的基因符号集合（用于匹配）
adata_symbols = set(pd.Series(gene_symbol).dropna().astype(str))

# =========================================================
# 3) 计算匹配率（全基因 / CellChatDB 基因 / ligand / receptor）
# =========================================================
n_total = len(gene_raw)
n_ens = int(is_ens.sum())
n_ens_mapped = int(mapped_success.sum())

# 你的基因符号中，有多少能在 CellChat 的 geneInfo Symbol 找到
n_in_cellchat = len(adata_symbols & cellchat_symbols)

# ligand 覆盖（你的基因里能覆盖多少配体）
n_ligand_hit = len(adata_symbols & ligands)

# receptor 覆盖
n_receptor_hit_raw = len(adata_symbols & receptors_raw)          # 保守（会偏低）
n_receptor_hit_sub = len(adata_symbols & receptor_subunits)      # 解析亚基（更合理）

# 也可以看：你的基因覆盖了多少“LR涉及到的基因集合”
lr_gene_set = set(ligands) | set(receptor_subunits)  # 用解析后的 receptor 亚基
n_lr_hit = len(adata_symbols & lr_gene_set)

# =========================================================
# 4) 打印“匹配率报告”
# =========================================================
def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b

print("\n================ CellChatDB 匹配率报告 ================")
print(f"adata 变量数（genes/features）: {n_total}")
print(f"其中 Ensembl 格式 (ENSG...) : {n_ens}  ({pct(n_ens, n_total):.2f}%)")
print(f"Ensembl 成功映射到 Symbol   : {n_ens_mapped} / {n_ens}  ({pct(n_ens_mapped, n_ens):.2f}%)")

print("\n[与 CellChatDB.geneInfo 的 Symbol 词表匹配]")
print(f"adata 中可被 CellChatDB 识别的 Symbol 数: {n_in_cellchat} / {len(adata_symbols)}  ({pct(n_in_cellchat, len(adata_symbols)):.2f}%)")

print("\n[LR 覆盖情况（基于 CellChatDB.interaction）]")
print(f"CellChatDB ligand 唯一基因数           : {len(ligands)}")
print(f"adata 覆盖到的 ligand 数               : {n_ligand_hit} / {len(ligands)}  ({pct(n_ligand_hit, len(ligands)):.2f}%)")

print(f"\nCellChatDB receptor(原始字段)唯一条目数 : {len(receptors_raw)}  （含复合体标签，偏“不可直接匹配”）")
print(f"adata 直接命中 receptor(原始字段)数    : {n_receptor_hit_raw} / {len(receptors_raw)}  ({pct(n_receptor_hit_raw, len(receptors_raw)):.2f}%)")

print(f"\n从 interaction_name_2 解析的 receptor 亚基基因数: {len(receptor_subunits)}")
print(f"adata 覆盖到的 receptor 亚基基因数            : {n_receptor_hit_sub} / {len(receptor_subunits)}  ({pct(n_receptor_hit_sub, len(receptor_subunits)):.2f}%)")

print(f"\nLR 基因全集（ligand ∪ receptor亚基）大小       : {len(lr_gene_set)}")
print(f"adata 覆盖到的 LR 基因数                        : {n_lr_hit} / {len(lr_gene_set)}  ({pct(n_lr_hit, len(lr_gene_set)):.2f}%)")
print("======================================================\n")








genes_raw = adata.var["gene"].astype(str).values
gene_list = genes_raw  # 这里直接用 gene 列（你报告显示大多数已是 Symbol）

gene_to_idx = {g: i for i, g in enumerate(gene_list)}

# -----------------------------
# 2) 准备 cluster 信息
# -----------------------------
celltype = adata.obs["cluster"].astype(str).values
types = pd.Categorical(celltype).categories.tolist()
n_types = len(types)

type_to_cells = {t: np.where(celltype == t)[0] for t in types}

# -----------------------------
# 3) 取出 LR 列表（只用 ligand / receptor）
#    注意：receptor 字段可能是复合体标签；这里用 interaction_name_2 解析括号内的受体亚基
# -----------------------------
def parse_receptor_subunits(s: str):
    if not isinstance(s, str):
        return []
    m = re.search(r"\((.*?)\)", s)
    if not m:
        return []
    inner = m.group(1)
    parts = [p.strip() for p in inner.split("+") if p.strip()]
    return parts

pairs = []
for _, row in interaction_df.iterrows():
    lig = str(row["ligand"])
    # receptor：优先解析 interaction_name_2 里的 (A+B+...)
    rec_sub = []
    if "interaction_name_2" in interaction_df.columns:
        rec_sub = parse_receptor_subunits(row.get("interaction_name_2", ""))
    if len(rec_sub) == 0:
        # 退化：直接用 receptor 字段（可能是复合体标签，命中率会低）
        rec_sub = [str(row["receptor"])]

    pairs.append((lig, rec_sub))

# 去重（同 ligand 可能对应多个 receptor 组合）
# pairs 结构：(ligand, [receptor_subunits...])
# -----------------------------
# 4) 预计算每个 cluster 的：
#    - mean expression（稀疏矩阵友好）
#    - pct expressed（>0 的比例）
# -----------------------------
X = adata.X
if not sp.issparse(X):
    X = sp.csr_matrix(X)

# mean expr per type: [n_types, n_genes]
mean_expr = np.zeros((n_types, X.shape[1]), dtype=np.float32)
pct_expr  = np.zeros((n_types, X.shape[1]), dtype=np.float32)

for ti, t in enumerate(types):
    idx = type_to_cells[t]
    Xsub = X[idx, :]
    # 平均
    mean_expr[ti, :] = np.asarray(Xsub.mean(axis=0)).ravel()
    # 表达比例（>0）
    # 对稀疏矩阵：每列非零个数 / n_cells
    nnz = np.diff(Xsub.tocsc().indptr)  # 每列 nnz
    pct_expr[ti, :] = nnz / max(len(idx), 1)

# -----------------------------
# 5) 计算 cluster->cluster 通讯强度矩阵
#    过滤策略（可调）：
#      - ligand 在 sender cluster: pct>=pct_th 且 mean>=mean_th
#      - receptor 每个亚基在 receiver cluster: pct>=pct_th_rec 且 mean>=mean_th_rec
#    强度定义：
#      score(L,Rset; A->B) = meanA(L) * geometric_mean(meanB(R_i))
#    最后对所有 pairs 求和
# -----------------------------
pct_th_lig = 0.05
pct_th_rec = 0.05
mean_th_lig = 0.0
mean_th_rec = 0.0

def geo_mean(vals):
    vals = np.asarray(vals, dtype=np.float32)
    vals = np.clip(vals, 1e-8, None)
    return float(np.exp(np.mean(np.log(vals))))

W = np.zeros((n_types, n_types), dtype=np.float64)

# 为了加速：先把“在数据中存在的基因”过滤
# ligand 必须存在于 gene_to_idx；receptor 亚基至少一个存在（或全部存在更严格）
for lig, rec_list in pairs:
    if lig not in gene_to_idx:
        continue
    lig_j = gene_to_idx[lig]

    # sender 侧 ligand 是否达标：不同 sender cluster 分别判断
    lig_mean_all = mean_expr[:, lig_j]
    lig_pct_all  = pct_expr[:, lig_j]

    # receptor 亚基索引
    rec_idx = [gene_to_idx[r] for r in rec_list if r in gene_to_idx]
    if len(rec_idx) == 0:
        continue

    # receiver 侧 receptor 亚基表达：不同 receiver cluster 分别判断
    rec_mean_all = mean_expr[:, rec_idx]  # [n_types, k]
    rec_pct_all  = pct_expr[:, rec_idx]

    for a in range(n_types):      # sender
        if lig_pct_all[a] < pct_th_lig or lig_mean_all[a] < mean_th_lig:
            continue

        lig_val = float(lig_mean_all[a])

        for b in range(n_types):  # receiver
            # receptor 亚基都要达标（更像 CellChat 对复合体的要求）
            if np.any(rec_pct_all[b] < pct_th_rec) or np.any(rec_mean_all[b] < mean_th_rec):
                continue

            rec_val = geo_mean(rec_mean_all[b])
            W[a, b] += lig_val * rec_val

# 归一化一下，让画图更稳定（可选）
# 这里用 log1p + min-max
W_log = np.log1p(W)
W_norm = (W_log - W_log.min()) / (W_log.max() - W_log.min() + 1e-12)

W_df = pd.DataFrame(W, index=types, columns=types)
W_df.to_csv("<CELLCHAT_OR_LR_OUTPUT_CSV>")        # 需改
print("<CELLCHAT_OR_LR_OUTPUT_CSV>")

# -----------------------------
# 6) 画 CellChat-like circle network（你给的那种风格）
#    - 节点大小：总入度+出度
#    - 边宽：W_norm[a,b]
#    - 自环：用弧线表示
# -----------------------------
def plot_cellchat_like_circle(
    Wn, labels,
    out_png="<OUTPUT_FIGURE_FILE>",
    topk_per_sender=3,          # 每个 sender 只保留最强的 k 条出边（强烈推荐）
    edge_min=0.12,              # 全局阈值（归一化后）
    node_scale=2600,
    edge_scale=18,
    node_alpha=0.95,
    edge_alpha=0.35,
    label_size=22,
    self_loop=True
):
    """
    Wn: [n,n] normalized matrix in [0,1]
    labels: list of node names (clusters)
    """

    n = len(labels)
    Wn = np.asarray(Wn, float)

    # ---- 1) 圆周布局（自己算，保证正圆）
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    R = 1.0
    pos = {labels[i]: np.array([R*np.cos(angles[i]), R*np.sin(angles[i])]) for i in range(n)}

    # ---- 2) 节点强度（入+出）
    strength = Wn.sum(axis=0) + Wn.sum(axis=1)
    strength = strength / (strength.max() + 1e-12)
    node_sizes = (0.25 + strength) * node_scale

    # ---- 3) 选边：每个 sender 取 topK（并且满足 edge_min）
    edges = []
    for i in range(n):
        row = Wn[i].copy()
        # 去掉太小的
        cand = [(i, j, row[j]) for j in range(n) if row[j] >= edge_min and j != i]
        cand.sort(key=lambda x: x[2], reverse=True)
        cand = cand[:topk_per_sender]
        edges.extend(cand)

    # 自环（可选）
    loops = []
    if self_loop:
        for i in range(n):
            if Wn[i, i] >= edge_min:
                loops.append((i, Wn[i, i]))

    # ---- 4) 颜色：按 sender 上色（CellChat 常用风格）
    cmap = cm.get_cmap("tab10" if n <= 10 else "tab20", n)
    node_colors = [cmap(i) for i in range(n)]
    sender_color = {i: node_colors[i] for i in range(n)}

    # ---- 5) 开始画
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect("equal")
    ax.axis("off")

    # 5.1 画边（先画细的，后画粗的，避免粗边盖住一切）
    edges.sort(key=lambda x: x[2])  # 从小到大画
    for i, j, w in edges:
        p1 = pos[labels[i]]
        p2 = pos[labels[j]]

        # 弧度：节点距离越远弧度越大；并且固定方向减少交叉
        dist = np.linalg.norm(p1 - p2)
        rad = 0.15 + 0.35*(dist/2.0)   # 0.15~0.5
        if i > j:
            rad = -rad

        lw = 0.3 + (np.sqrt(w) * edge_scale)  # sqrt 缩放更好看
        col = sender_color[i]

        arrow = FancyArrowPatch(
            p1, p2,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=col,
            alpha=edge_alpha
        )
        ax.add_patch(arrow)

    # 5.2 自环
    for i, w in loops:
        p = pos[labels[i]]
        lw = 0.3 + (np.sqrt(w) * edge_scale)
        col = sender_color[i]
        loop = Arc(
            (p[0], p[1]),
            0.35, 0.35,     # loop size
            angle=0,
            theta1=25, theta2=340,
            linewidth=lw,
            color=col,
            alpha=edge_alpha
        )
        ax.add_patch(loop)

    # 5.3 画节点
    for i, lab in enumerate(labels):
        p = pos[lab]
        ax.scatter([p[0]], [p[1]], s=node_sizes[i], color=node_colors[i], alpha=node_alpha, edgecolors="none")

    # 5.4 标签：外移 + 旋转（像 CellChat）
    for i, lab in enumerate(labels):
        p = pos[lab]
        # 外移距离
        text_r = 1.25
        tx, ty = text_r*np.cos(angles[i]), text_r*np.sin(angles[i])

        # 旋转角，让标签沿圆周方向
        rot = np.degrees(angles[i])
        # 左半边翻转，保证正向可读
        ha = "left"
        if 90 < rot < 270:
            rot += 180
            ha = "right"

        ax.text(tx, ty, lab, fontsize=label_size, rotation=rot, rotation_mode="anchor",
                ha=ha, va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out_png)


plot_cellchat_like_circle(
    W_norm, types,
    out_png="<OUTPUT_FIGURE_FILE>",  # 需改
    topk_per_sender=3,
    edge_min=0.18,
    edge_scale=18,
    label_size=20
)







"""
#print(interaction_df.head())
#print(interaction_df.columns)


interaction_df.to_csv(
    "<CELLCHAT_INTERACTION_CSV>",
    index=False
)
"""