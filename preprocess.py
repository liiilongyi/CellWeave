from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from anndata import AnnData
from utils import load_tissue_hires_scalef

def inspect_h5(h5_path: Path, max_show: int = 10):
    """Print a compact tree of groups/datasets and detect likely 10x layouts."""
    import h5py
    if not h5_path.exists():
        print(f"[inspect] Missing: {h5_path}")
        return
    with h5py.File(h5_path, 'r') as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = str(obj.dtype)
                print(f"[DSET] /{name} shape={shape} dtype={dtype}")
            else:
                print(f"[GRP ] /{name}")
        print(f"===== Inspecting {h5_path.name} =====")
        f.visititems(walk)
        print("==================================")

def _load_10x_h5_relaxed(h5_path: Path):
    """Permissive loader for 10x-style H5 that handles CSC layout.
    10x stores matrix as **CSC**: data, indices(=row idx), indptr(=col ptr),
    shape=(n_rows=genes, n_cols=barcodes). We convert to CSR with barcodes as rows.
    Returns: X(csr: n_barcodes×n_genes), gene_names(np.array[str]), barcodes(np.array[str])
    """
    import h5py
    from scipy import sparse
    with h5py.File(h5_path, 'r') as f:
        def _as_unicode(arr):
            # handle fixed-length bytes strings like |S21
            if hasattr(arr, 'astype'):
                try:
                    return arr.astype('U')
                except Exception:
                    return np.array([x.decode('utf-8', 'ignore') for x in arr])
            return np.array(arr, dtype='U')

        if 'matrix' in f:
            grp = f['matrix']
            data = grp['data'][:]
            indices = grp['indices'][:]
            indptr = grp['indptr'][:]
            shape = tuple(grp['shape'][:])  # (n_genes, n_barcodes)

            # Build CSC first (10x convention), then transpose to CSR so that rows=barcodes
            X_csc = sparse.csc_matrix((data, indices, indptr), shape=shape)
            X_csr = X_csc.T.tocsr()

            # features names
            feat_group = grp.get('features')
            if feat_group is not None:
                if 'name' in feat_group:
                    gene_names = _as_unicode(feat_group['name'][:])
                elif 'gene_names' in feat_group:
                    gene_names = _as_unicode(feat_group['gene_names'][:])
                else:
                    cand = [k for k in feat_group.keys() if 'name' in k.lower()]
                    if cand:
                        gene_names = _as_unicode(feat_group[cand[0]][:])
                    else:
                        raise ValueError("could not find feature names in features group")
            else:
                for k in ['gene_names', 'features']:
                    if k in f:
                        gene_names = _as_unicode(f[k][:])
                        break
                else:
                    raise ValueError("no features group or gene_names at root")

            # barcodes
            if 'barcodes' in grp:
                barcodes = _as_unicode(grp['barcodes'][:])
            elif 'obs' in grp and 'barcodes' in grp['obs']:
                barcodes = _as_unicode(grp['obs']['barcodes'][:])
            elif 'barcodes' in f:
                barcodes = _as_unicode(f['barcodes'][:])
            else:
                cand = []
                def _collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and 'barcode' in name.lower():
                        cand.append('/' + name)
                f.visititems(_collect)
                if not cand:
                    raise ValueError("could not find barcodes dataset")
                barcodes = _as_unicode(f[cand[0]][:])

            # Sanity: X rows should match #barcodes, cols should match #genes
            if X_csr.shape[0] != len(barcodes) or X_csr.shape[1] != len(gene_names):
                raise ValueError(
                    f"Dimension mismatch after CSC→CSR: X={X_csr.shape}, barcodes={len(barcodes)}, genes={len(gene_names)}")
            return X_csr, gene_names, barcodes

        # Older layout at root (assume CSC as well)
        needed = ['data', 'indices', 'indptr', 'shape']
        if all(k in f for k in needed):
            shape = tuple(f['shape'][:])
            X_csc = sparse.csc_matrix((f['data'][:], f['indices'][:], f['indptr'][:]), shape=shape)
            X_csr = X_csc.T.tocsr()
            # gene names
            for k in ['gene_names', 'genes', 'features']:
                if k in f:
                    gene_names = _as_unicode(f[k][:])
                    break
            else:
                raise ValueError("no gene names dataset found")
            # barcodes
            for k in ['barcodes', 'obs']:
                if k in f:
                    barcodes = _as_unicode(f[k][:])
                    break
            else:
                raise ValueError("no barcodes dataset found")
            if X_csr.shape[0] != len(barcodes) or X_csr.shape[1] != len(gene_names):
                raise ValueError(
                    f"Dimension mismatch after CSC→CSR: X={X_csr.shape}, barcodes={len(barcodes)}, genes={len(gene_names)}")
            return X_csr, gene_names, barcodes

        raise ValueError("Unrecognized 10x h5 structure; check inspect_h5 output.")

def main():
    root = Path(r"<LOCAL_DATA_OR_OUTPUT_PATH>")
    use_filtered_first = True        # prefer filtered over raw

    spatial_dir = root / 'spatial'
    img_path = spatial_dir / '<H_AND_E_IMAGE_FILE>'
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    # Decide which h5 to use, but also *inspect* both and sanity-check shapes
    f_filtered = root / '<TENX_FEATURE_MATRIX_H5>'
    f_raw = root / '<TENX_FEATURE_MATRIX_H5>'

    # Optional: print quick info to console so you can verify content & sizes
    if f_filtered.exists():
        inspect_h5(f_filtered)
    if f_raw.exists():
        inspect_h5(f_raw)

    # Choose path according to availability and flag
    candidates = []
    if use_filtered_first and f_filtered.exists():
        candidates = [f_filtered, f_raw]
    else:
        candidates = [f_raw, f_filtered]

    chosen = None
    X = None; gene_names = None; barcodes = None
    for h5 in candidates:
        if h5 is None or not h5.exists():
            continue
        try:
            X_tmp, G_tmp, B_tmp = _load_10x_h5_relaxed(h5)
            # Basic sanity: at least 1k bins and 100 genes (tune to your data)
            if X_tmp.shape[0] >= 100 and X_tmp.shape[1] >= 100:
                chosen = h5; X = X_tmp; gene_names = G_tmp; barcodes = B_tmp
                break
        except Exception as e:
            print(f"[warn] Failed to parse {h5.name}: {e}")
    if chosen is None:
        raise RuntimeError("Could not parse either filtered or raw h5; see inspection logs above.")
    print(f"Using matrix: {chosen.name}  shape={X.shape}")

    print("\n===> [1] Loading positions …")
    # positions（固定映射：X=pxl_col_in_fullres, Y=pxl_row_in_fullres）
    pos_path = spatial_dir / '<VISIUM_TISSUE_POSITIONS_PARQUET>'
    pos_raw = pd.read_parquet(pos_path)

    print(f"     Loaded positions: {len(pos_raw)} rows")

    # 只取需要的列，并把像素列改名为 px/py，便于后续使用
    pos = pos_raw[['barcode', 'in_tissue', 'array_row', 'array_col',
                   'pxl_row_in_fullres', 'pxl_col_in_fullres']].copy()


    pos.rename(columns={'pxl_col_in_fullres': 'px',  # X = col
                        'pxl_row_in_fullres': 'py'},  # Y = row
               inplace=True)
    pos['in_tissue'] = pos['in_tissue'].astype(int)

    print("Number of bins in tissue:", (pos["in_tissue"] == 1).sum())
    pos_in = pos[pos["in_tissue"] == 1]

    he_path = spatial_dir / "<H_AND_E_IMAGE_FILE>"

    # 读图 & 图像尺寸（PIL: size = (宽W, 高H)）
    img = Image.open(he_path).convert("RGB")
    W, H = img.size # (6000, 5492)
    px = pos_in["px"].to_numpy()
    py = pos_in["py"].to_numpy()

    scalef_json = spatial_dir / "<VISIUM_SCALEFACTORS_JSON>"
    s = load_tissue_hires_scalef(scalef_json)
    print("s = ", s)
    pxn = px * s
    pyn = py * s

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.scatter(pxn, pyn, s=0.01, alpha=0.1)  # s 是点的面积（像素^2），数值越小点越小
    plt.gca().set_xlim(0, W)
    plt.gca().set_ylim(H, 0)  # 像素坐标 y 轴向下
    plt.gca().set_aspect('equal', 'box')  # 长宽比一致
    plt.axis('off')
    plt.savefig("<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
    plt.show()

    print("H&E size (W,H):", W, H)
    print("px range:", px.min(), px.max())
    print("py range:", py.min(), py.max())


    print("\n===> [2] Aligning barcodes to matrix indices …")

    # 过滤 in_tissue 并与矩阵对齐
    bc2idx = {b: i for i, b in enumerate(barcodes)}
    pos_in = pos[pos["in_tissue"] == 1].copy()
    pos_in = pos_in[pos_in["barcode"].isin(bc2idx)]

    node_barcodes = pos_in["barcode"].to_numpy()
    node_index = np.array([bc2idx[b] for b in node_barcodes], dtype=np.int64)
    Xk = X[node_index]

    print("Filtered expression matrix shape:", Xk.shape)
    print("Matrix density:", Xk.nnz / (Xk.shape[0] * Xk.shape[1]))
    for i in range(5):
        row = Xk.getrow(i)
        genes_nonzero = np.array(gene_names)[row.indices]
        print(f"\n=== Bin {i} ({pos_in.iloc[i]['barcode']}) ===")
        print(f"Nonzero genes: {len(genes_nonzero)}")
        print(pd.DataFrame({
            "gene": genes_nonzero,
            "expr": row.data
        }).sort_values("expr", ascending=False).head(10))
    nonzero_per_bin = np.diff(Xk.indptr)
    print("每个 bin 平均非零基因数:", nonzero_per_bin.mean())
    print("最小:", nonzero_per_bin.min(), "最大:", nonzero_per_bin.max())

    # 每行的非零元素数（与前面一致）
    nonzero_per_bin = np.diff(Xk.indptr)

    # 找出全 0 的 bin
    zero_bins = np.where(nonzero_per_bin == 0)[0]

    print(f"Total bins: {Xk.shape[0]}")
    print(f"Bins with all-zero expression: {len(zero_bins)}")
    print(f"Percentage all-zero: {len(zero_bins) / Xk.shape[0] * 100:.2f}%")

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    # 非零 bin
    mask_nonzero = np.ones_like(pxn, dtype=bool)
    mask_nonzero[zero_bins] = False
    plt.scatter(pxn[mask_nonzero], pyn[mask_nonzero], s=0.01, alpha=0.1, color='gray')
    # 全零 bin（用红色标出）
    plt.scatter(pxn[zero_bins], pyn[zero_bins], s=0.3, alpha=0.3, color='red')
    plt.gca().set_xlim(0, W)
    plt.gca().set_ylim(H, 0)  # 图像 y 轴向下
    plt.gca().set_aspect('equal', 'box')
    plt.axis('off')
    plt.legend(loc='upper right', markerscale=20, frameon=False)
    plt.savefig("<OUTPUT_FIGURE_FILE>", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Total bins: {len(pxn)} | All-zero bins: {len(zero_bins)} ({len(zero_bins) / len(pxn) * 100:.2f}%)")

    print('making h5ad...')
    pos_in_aligned = pos_in.reset_index(drop=True)

    # ---- obs ----
    obs = pd.DataFrame({
        "barcode": pos_in_aligned["barcode"].values,
        "px": pos_in_aligned["px"].values,
        "py": pos_in_aligned["py"].values,
    })
    obs.index = obs["barcode"]  # 用 barcode 当索引
    obs.drop(columns=["barcode"], inplace=True)

    # ---- var ----
    var = pd.DataFrame({
        "gene": gene_names
    })

    # ---- AnnData ----
    adata = AnnData(
        X=Xk,  # 稀疏矩阵直接放
        obs=obs,
        var=var
    )

    print(adata)
    print(adata.obs['px'])
    print(adata.obs['py'])

    # ---- 保存 ----
    out_path = "<BIN_EXPRESSION_H5AD>"
    adata.write_h5ad(out_path)

    print("Saved h5ad:", out_path)
    print(adata)


if __name__ == '__main__':
    # Direct call; parameters are set at the top of main()
    main()
