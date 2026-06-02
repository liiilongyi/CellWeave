import json
import numpy as np
from pathlib import Path

def filter_bins(adata, vocab_json_path="<SCGPT_VOCAB_JSON>"):
    """
    输入:
        adata: AnnData 对象 (bin × gene)
        vocab_json_path: scGPT 词表文件路径 (默认 vocab.json)

    返回:
        过滤后的 AnnData
    """

    # ===== ① 删除全零 bin =====
    nnz_per_bin = np.diff(adata.X.indptr)
    keep = nnz_per_bin > 0

    print(f"Total bins: {adata.n_obs}")
    print(f"Zero-expression bins: {(~keep).sum()}")
    print(f"Remaining bins after removing zero bins: {keep.sum()}")

    adata = adata[keep].copy()
    print("After filtering zero bins:", adata)

    # ===== ② 载入 scGPT 词表 =====
    with open(vocab_json_path, "r") as f:
        vocab = json.load(f)

    # 过滤掉不是基因的 token（如 <pad> <cls>）
    vocab_genes = {k for k in vocab.keys() if not k.startswith("<")}

    # ===== ③ 基因名大写化后与词表交集 =====
    adata.var["gene"] = adata.var["gene"].astype(str).str.upper()
    genes = adata.var["gene"].astype(str)

    keep_gene_mask = genes.isin(vocab_genes)

    print("原始基因数:", adata.n_vars)
    print("与 scGPT 词表交集的基因数:", keep_gene_mask.sum())

    adata = adata[:, keep_gene_mask].copy()

    # ===== ④ 删除在子集矩阵中仍然全 0 的 bin =====
    X = adata.X
    cell_sum = np.array(X.sum(axis=1)).ravel()
    nonzero_cell_mask = cell_sum > 0

    print("删除全零基因后剩余 bin 数:", adata.n_obs)
    print("仍有表达的 bin 数:", nonzero_cell_mask.sum())

    adata = adata[nonzero_cell_mask].copy()
    print("Final filtered adata:", adata)

    return adata

def filter_bins_keep_genes(adata, vocab_json_path="<SCGPT_VOCAB_JSON>"):
    """
    输入:
        adata: AnnData 对象 (bin × gene)
        vocab_json_path: scGPT 词表文件路径 (默认 vocab.json)

    返回:
        仅在 bin 维度做过滤、不删减基因的 AnnData
    """

    # ===== ① 删除原始矩阵中全零 bin =====
    # 这里假设 adata.X 是 csr_matrix
    nnz_per_bin = np.diff(adata.X.indptr)
    keep_bin_mask = nnz_per_bin > 0

    print(f"Total bins: {adata.n_obs}")
    print(f"Zero-expression bins: {(~keep_bin_mask).sum()}")
    print(f"Remaining bins after removing zero bins: {keep_bin_mask.sum()}")

    adata = adata[keep_bin_mask].copy()
    print("After filtering zero bins:", adata)

    # ===== ② 载入 scGPT 词表 =====
    with open(vocab_json_path, "r") as f:
        vocab = json.load(f)

    # 过滤掉不是基因的 token（如 <pad> <cls>）
    vocab_genes = {k for k in vocab.keys() if not k.startswith("<")}

    # ===== ③ 基因名大写化（但不删基因，只做统计用） =====
    adata.var["gene"] = adata.var["gene"].astype(str).str.upper()
    #adata.var["gene"] = adata.var.index.astype(str).str.upper()
    genes = adata.var["gene"].astype(str)

    # 与 scGPT 词表的交集掩码（仅用于后面判断哪些 bin 该删）
    keep_gene_mask = genes.isin(vocab_genes)

    print("原始基因数:", adata.n_vars)
    print("与 scGPT 词表交集的基因数:", keep_gene_mask.sum())
    # 注意：这里不再对 adata 在基因维度做切片，即不删基因

    # ===== ④ 再次删除“在词表基因上全 0”的 bin（保持原来删 bin 的逻辑）=====
    X = adata.X
    # 只在与词表交集的基因上做求和，用于判断哪些 bin 在这些基因上全 0
    # 但不会真的删掉这些基因列
    gene_mask_np = keep_gene_mask.to_numpy() if hasattr(keep_gene_mask, "to_numpy") else np.array(keep_gene_mask)
    if gene_mask_np.sum() > 0:
        cell_sum_on_vocab_genes = np.array(X[:, gene_mask_np].sum(axis=1)).ravel()
    else:
        # 如果词表与基因完全没有交集，则所有 bin 在“词表基因”上都是 0
        cell_sum_on_vocab_genes = np.zeros(adata.n_obs, dtype=float)

    nonzero_cell_mask = cell_sum_on_vocab_genes > 0

    print("删除全零基因后剩余 bin 数（逻辑上等价于原函数的第二次删 bin）:", adata.n_obs)
    print("在词表基因上仍有表达的 bin 数:", nonzero_cell_mask.sum())

    adata = adata[nonzero_cell_mask].copy()
    print("Final filtered adata (bins filtered, genes kept):", adata)

    return adata

def load_tissue_hires_scalef(json_path: Path) -> float:
    """
    从 10x 的 scalefactors_json.json 中读取 tissue_hires_scalef。
    参数:
        json_path: scalefactors_json.json 的完整路径
    返回:
        tissue_hires_scalef (float)
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    if "tissue_hires_scalef" not in data:
        raise KeyError(f"'tissue_hires_scalef' not found in {json_path}")
    return float(data["tissue_hires_scalef"])
