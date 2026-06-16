import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from anndata import AnnData

# ==================== Local Definition: Replaces external utils import to prevent naming conflicts ====================
def load_tissue_hires_scalef(scalef_json_path):
    """Read official 10x scaling factors configuration file"""
    import json
    with open(scalef_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Extract high-resolution scaling factor, default value is 1.0
    return data.get("tissue_hires_scalef", 1.0)
# ==================================================================================================================


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
    """Permissive loader for 10x-style H5 that handles CSC layout."""
    import h5py
    from scipy import sparse
    with h5py.File(h5_path, 'r') as f:
        def _as_unicode(arr):
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
            shape = tuple(grp['shape'][:])

            X_csc = sparse.csc_matrix((data, indices, indptr), shape=shape)
            X_csr = X_csc.T.tocsr()

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

            if X_csr.shape[0] != len(barcodes) or X_csr.shape[1] != len(gene_names):
                raise ValueError(
                    f"Dimension mismatch after CSC→CSR: X={X_csr.shape}, barcodes={len(barcodes)}, genes={len(gene_names)}")
            return X_csr, gene_names, barcodes

        needed = ['data', 'indices', 'indptr', 'shape']
        if all(k in f for k in needed):
            shape = tuple(f['shape'][:])
            X_csc = sparse.csc_matrix((f['data'][:], f['indices'][:], f['indptr'][:]), shape=shape)
            X_csr = X_csc.T.tocsr()
            for k in ['gene_names', 'genes', 'features']:
                if k in f:
                    gene_names = _as_unicode(f[k][:])
                    break
            else:
                raise ValueError("no gene names dataset found")
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
    # ==================== Dynamically Read Global Configuration ====================
    config_path = Path("pipeline_config.json")
    sample_id = "mouse_brain"
    pos_path_str = "binned_outputs/square_002um/spatial/tissue_positions.parquet"
    f_filtered_str = "binned_outputs/square_002um/filtered_feature_bc_matrix.h5"
    f_raw_str = "binned_outputs/square_002um/raw_feature_bc_matrix.h5"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        sample_id = cfg.get("SAMPLE_ID", "mouse_brain")
        pos_path_str = cfg.get("POS_PATH", pos_path_str)
        f_filtered_str = cfg.get("EXPR_FILTERED_H5", f_filtered_str)
        f_raw_str = cfg.get("EXPR_RAW_H5", f_raw_str)

    pos_path = Path(pos_path_str)
    f_filtered = Path(f_filtered_str)
    f_raw = Path(f_raw_str)

    # Dynamically infer the location of the spatial folder
    spatial_dir = pos_path.parent
    # ==============================================================================

    use_filtered_first = True  # Prioritize selecting the filtered expression matrix

    if f_filtered.exists():
        inspect_h5(f_filtered)
    if f_raw.exists():
        inspect_h5(f_raw)

    candidates = []
    if use_filtered_first and f_filtered.exists():
        candidates = [f_filtered, f_raw]
    else:
        candidates = [f_raw, f_filtered]

    chosen = None
    X = None;
    gene_names = None;
    barcodes = None
    for h5 in candidates:
        if h5 is None or not h5.exists():
            continue
        try:
            X_tmp, G_tmp, B_tmp = _load_10x_h5_relaxed(h5)
            if X_tmp.shape[0] >= 100 and X_tmp.shape[1] >= 100:
                chosen = h5;
                X = X_tmp;
                gene_names = G_tmp;
                barcodes = B_tmp
                break
        except Exception as e:
            print(f"[warn] Failed to parse {h5.name}: {e}")
    if chosen is None:
        raise RuntimeError("Could not parse either filtered or raw h5; see inspection logs.")
    print(f"Using matrix: {chosen.name}  shape={X.shape}")

    print("\n===> [1] Loading positions …")
    pos_raw = pd.read_parquet(pos_path)
    print(f"     Loaded positions: {len(pos_raw)} rows")

    pos = pos_raw[['barcode', 'in_tissue', 'array_row', 'array_col',
                   'pxl_row_in_fullres', 'pxl_col_in_fullres']].copy()

    pos.rename(columns={'pxl_col_in_fullres': 'px',  # X = col
                        'pxl_row_in_fullres': 'py'},  # Y = row
               inplace=True)
    pos['in_tissue'] = pos['in_tissue'].astype(int)

    print("Number of bins in tissue:", (pos["in_tissue"] == 1).sum())
    pos_in = pos[pos["in_tissue"] == 1]

    he_path = spatial_dir / "tissue_hires_image.png"

    img = Image.open(he_path).convert("RGB")
    W, H = img.size
    px = pos_in["px"].to_numpy()
    py = pos_in["py"].to_numpy()

    scalef_json = spatial_dir / "scalefactors_json.json"
    s = load_tissue_hires_scalef(scalef_json)
    print("s = ", s)
    pxn = px * s
    pyn = py * s

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.scatter(pxn, pyn, s=0.01, alpha=0.1)
    plt.gca().set_xlim(0, W)
    plt.gca().set_ylim(H, 0)
    plt.gca().set_aspect('equal', 'box')
    plt.axis('off')

    os.makedirs("figure", exist_ok=True)
    plt.savefig(f"figure/{sample_id}_visium_bins_all.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("H&E size (W,H):", W, H)
    print("px range:", px.min(), px.max())
    print("py range:", py.min(), py.max())

    print("\n===> [2] Aligning barcodes to matrix indices …")

    bc2idx = {b: i for i, b in enumerate(barcodes)}
    pos_in = pos[pos["in_tissue"] == 1].copy()
    pos_in = pos_in[pos_in["barcode"].isin(bc2idx)]

    node_barcodes = pos_in["barcode"].to_numpy()
    node_index = np.array([bc2idx[b] for b in node_barcodes], dtype=np.int64)
    Xk = X[node_index]

    print("Filtered expression matrix shape:", Xk.shape)
    print("Matrix density:", Xk.nnz / (Xk.shape[0] * Xk.shape[1]))

    nonzero_per_bin = np.diff(Xk.indptr)
    print("Average non-zero genes per bin:", nonzero_per_bin.mean())
    print("Min:", nonzero_per_bin.min(), "Max:", nonzero_per_bin.max())

    zero_bins = np.where(nonzero_per_bin == 0)[0]

    print(f"Total bins: {Xk.shape[0]}")
    print(f"Bins with all-zero expression: {len(zero_bins)}")
    print(f"Percentage all-zero: {len(zero_bins) / Xk.shape[0] * 100:.2f}%")

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    mask_nonzero = np.ones_like(pxn, dtype=bool)
    mask_nonzero[mask_nonzero == True] = True # Default structural mask placeholder
    mask_nonzero[zero_bins] = False
    plt.scatter(pxn[mask_nonzero], pyn[mask_nonzero], s=0.01, alpha=0.1, color='gray')
    plt.scatter(pxn[zero_bins], pyn[zero_bins], s=0.3, alpha=0.3, color='red')
    plt.gca().set_xlim(0, W)
    plt.gca().set_ylim(H, 0)
    plt.gca().set_aspect('equal', 'box')
    plt.axis('off')
    plt.savefig(f"figure/{sample_id}_visium_bins_zero_vs_nonzero.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print('making h5ad...')
    pos_in_aligned = pos_in.reset_index(drop=True)

    # ---- obs ----
    obs = pd.DataFrame({
        "barcode": pos_in_aligned["barcode"].values,
        "px": pos_in_aligned["px"].values,
        "py": pos_in_aligned["py"].values,
    })
    obs.index = obs["barcode"]
    obs.drop(columns=["barcode"], inplace=True)

    # ---- var ----
    var = pd.DataFrame({
        "gene": gene_names
    })

    # ---- AnnData ----
    adata = AnnData(
        X=Xk,
        obs=obs,
        var=var
    )

    # ---- Save with dynamic sample name ----
    out_path = f"{sample_id}_all.h5ad"
    adata.write_h5ad(out_path)

    print("Saved h5ad successfully:", out_path)


if __name__ == '__main__':
    main()
