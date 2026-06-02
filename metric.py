import numpy as np
import pandas as pd
import scanpy as sc
import numpy as np
import pandas as pd
import scanpy as s
def compute_iou_matching_from_bin_labels(adata, pred_col: str, gt_parquet_path: str, gt_barcode_col: str = "barcode",gt_cellid_col: str = "cell_id",gt_background: int = -1, pred_background: int = 0, iou_threshold: float = 0.3):
    """
    通过 bin-level GT/Pred label 计算实例分割 IoU，并做一对一匹配，输出 mIoU / F1 / PQ 等指标。
    【优化版】：彻底解决了海量细胞时的内存核弹问题，使用 Pandas 稀疏哈希聚合。
    """

    # ========== 1) 读取 GT，并对齐到 adata.obs_names ==========
    gt_df = pd.read_parquet(gt_parquet_path, columns=[gt_barcode_col, gt_cellid_col])
    gt_df = gt_df.drop_duplicates(subset=[gt_barcode_col]).set_index(gt_barcode_col)

    # 对齐顺序到 adata
    barcodes = adata.obs_names
    gt_series = gt_df.reindex(barcodes)[gt_cellid_col]

    if gt_series.isna().any():
        n_missing = int(gt_series.isna().sum())
        raise ValueError(f"GT parquet 里有 {n_missing} 个 barcode 无法在 adata.obs_names 中对齐，请检查 barcode 一致性。")

    gt_ids = gt_series.to_numpy(dtype=np.int64)
    pred_ids = adata.obs[pred_col].to_numpy(dtype=np.int64)

    # ========== 2) 过滤背景 bin ==========
    valid = (gt_ids != gt_background) & (pred_ids != pred_background)
    gt_ids_v = gt_ids[valid]
    pred_ids_v = pred_ids[valid]

    if gt_ids_v.size == 0:
        raise ValueError("过滤背景后没有任何有效 bin，请检查 gt_background/pred_background 设置是否正确。")

    # ========== 3) Factorize ==========
    gt_codes, gt_uniques = pd.factorize(gt_ids_v, sort=True)
    pred_codes, pred_uniques = pd.factorize(pred_ids_v, sort=True)

    n_gt = len(gt_uniques)
    n_pred = len(pred_uniques)

    # area 统计
    area_gt = np.bincount(gt_codes, minlength=n_gt).astype(np.int64)
    area_pred = np.bincount(pred_codes, minlength=n_pred).astype(np.int64)

    # ========== 4) 统计交集 (🚨 完美修复内存核弹) ==========
    df_inter = pd.DataFrame({'g': gt_codes, 'p': pred_codes})
    inter_counts = df_inter.groupby(['g', 'p'], sort=False).size().reset_index(name='intersection')

    gt_nz = inter_counts['g'].values
    pred_nz = inter_counts['p'].values
    inter = inter_counts['intersection'].values

    # ========== 5) 计算每个候选对的 IoU ==========
    union = area_gt[gt_nz] + area_pred[pred_nz] - inter
    iou = inter / np.maximum(union, 1)

    # ========== 6) 一对一匹配（Greedy by IoU） ==========
    keep = iou >= iou_threshold
    gt_nz2 = gt_nz[keep]
    pred_nz2 = pred_nz[keep]
    inter2 = inter[keep]
    iou2 = iou[keep]

    total = gt_codes.size
    if iou2.size == 0:
        metrics = {
            "iou_threshold": iou_threshold, "TP": 0, "FP": int(n_pred), "FN": int(n_gt),
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "mIoU_matched": 0.0,
            "PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "n_gt": int(n_gt), "n_pred": int(n_pred),
            "n_valid_bins": int(total),
        }
        matches_df = pd.DataFrame(columns=["gt_id", "pred_id", "intersection_bins", "iou"])
        return matches_df, metrics

    sort_idx = np.argsort(-iou2)
    gt_nz2 = gt_nz2[sort_idx]
    pred_nz2 = pred_nz2[sort_idx]
    inter2 = inter2[sort_idx]
    iou2 = iou2[sort_idx]

    used_gt = np.zeros(n_gt, dtype=bool)
    used_pred = np.zeros(n_pred, dtype=bool)

    matched_gt, matched_pred, matched_inter, matched_iou = [], [], [], []

    for g, p, it, v in zip(gt_nz2, pred_nz2, inter2, iou2):
        if (not used_gt[g]) and (not used_pred[p]):
            used_gt[g] = True
            used_pred[p] = True
            matched_gt.append(g)
            matched_pred.append(p)
            matched_inter.append(it)
            matched_iou.append(v)

    matched_gt = np.array(matched_gt, dtype=np.int64)
    matched_pred = np.array(matched_pred, dtype=np.int64)
    matched_inter = np.array(matched_inter, dtype=np.int64)
    matched_iou = np.array(matched_iou, dtype=np.float64)

    # ========== 7) 还原成原始 cell id，并计算指标 ==========
    gt_id_out = gt_uniques[matched_gt]
    pred_id_out = pred_uniques[matched_pred]

    TP = matched_iou.size
    FP = int(n_pred - TP)
    FN = int(n_gt - TP)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mIoU = float(matched_iou.mean()) if TP > 0 else 0.0

    SQ = mIoU
    RQ = TP / (TP + 0.5 * FP + 0.5 * FN) if (TP + FP + FN) > 0 else 0.0
    PQ = SQ * RQ

    matches_df = pd.DataFrame({
        "gt_id": gt_id_out, "pred_id": pred_id_out,
        "intersection_bins": matched_inter, "iou": matched_iou,
    }).sort_values("iou", ascending=False)

    metrics = {
        "iou_threshold": float(iou_threshold), "TP": int(TP), "FP": int(FP), "FN": int(FN),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mIoU_matched": float(mIoU), "SQ": float(SQ), "RQ": float(RQ), "PQ": float(PQ),
        "n_gt": int(n_gt), "n_pred": int(n_pred), "n_valid_bins": int(total),
    }
    return matches_df, metrics

if __name__ == "__main__":

    #adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>", backed='r')
    #adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>", backed='r')
    # adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>", backed='r')
    adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>", backed='r')

    #adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>", backed='r')
    #adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>", backed='r') # SCS

    pred_col = "<PREDICTED_SEGMENTATION_COLUMN>"
    # pred_col = "l1_h64_12_ear_630_7808"
    #pred_col = "seed_STP"
    # pred_col = "<PREDICTED_SEGMENTATION_COLUMN>"
    #gt_path = r"<GROUND_TRUTH_CELL_LABELS_PARQUET>"  # mouse brain
    gt_path = r"<GROUND_TRUTH_CELL_LABELS_PARQUET>"       # Ovarian


    matches, metrics = compute_iou_matching_from_bin_labels(
        adata,
        pred_col=pred_col,
        gt_parquet_path=gt_path,
        iou_threshold=0.3
    )
print(metrics)
print(matches.head(5))
print(pred_col)

exit()

"""
只出现一次的seed赋值为0
计数
"""

src_col = 'l3_cut13_ear_e1160_8945'
dst_col = 'l3_cut13_ear_e1160_8945_minus'

adata.obs[dst_col] = adata.obs[src_col].copy()
seed_counts = adata.obs[src_col].value_counts()
single_seeds = seed_counts[(seed_counts == 1) & (seed_counts.index != 0)].index
adata.obs.loc[
    adata.obs[dst_col].isin(single_seeds),
    dst_col
] = 0
print(adata)


seed = adata.obs['l3_cut13_ear_e1160_8945_minus']
seed = seed[seed != 0]

# 1️⃣ 统计每个 seed 出现的次数（= 每个细胞包含的 bin 数）
seed_counts = seed.value_counts()

# 2️⃣ 统计只出现 1–10 次的 seed 数量
for i in range(1, 11):
    cnt = (seed_counts == i).sum()
    print(f"只出现 {i} 次的 seed 数量: {cnt}")

# （可选）顺带看一下总共有多少个 seed
print(f"总 seed 数量（不含背景）: {seed_counts.shape[0]}")


exit()
