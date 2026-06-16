import os
import numpy as np
import scanpy as sc
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree

config_path = "pipeline_config.json"
sample_id = "mouse_brain_demo"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sample_id = cfg.get("SAMPLE_ID", "mouse_brain_demo")

# Dynamically locate the full map and model features generated in previous steps
adata = sc.read_h5ad(f"{sample_id}_seed.h5ad")
data = np.load(f"weight/{sample_id}/demo.npz", allow_pickle=True)
all_h5ad_path = f"{sample_id}_seed.h5ad"  # <--- Key modification: Target the raw, unfiltered full-volume h5ad after preprocessing

# Dynamically read the corresponding graph model inference results
data = np.load(f"weight/{sample_id}/demo.npz", allow_pickle=True)

def summarize_seed_stats(adata, key):
    """
    Input: adata (AnnData)
    Function: Calculate basic statistics for the seed column
    Including:
        1. Number of non- -1 values
        2. Number of 0 values
        3. Number of positive integer (>0) categories
        4. Average occurrences per positive integer
    """
    seed = adata.obs[key].to_numpy()

    # 2. Count of 0 occurrences
    count_zero = np.sum(seed == 0)
    # 3. Count of unique positive integer (>0) categories
    valid_labels = seed[seed > 0]
    num_unique_labels = len(np.unique(valid_labels))
    # 4. Average occurrences per positive integer
    avg_per_label = len(valid_labels) / num_unique_labels if num_unique_labels > 0 else 0
    print("Number of zeros =", count_zero)
    print("Number of unique positive labels =", num_unique_labels)
    print("Average occurrences per positive label =", avg_per_label)


def visualize_msf_mask_with_seeds(adata, mask_key="cell_id_msf", seed_key="seed", figsize=(6, 6), output_path=None,
                                  show=True):
    """
    Display based on bin coordinates (parsed from barcodes):
    - mask_key != 0 → White
    - mask_key == 0 → Black
    - seed_key > 0 → Red (overlay on top)
    Output as an RGB image.
    """

    # ===== 1. Parse coordinates from barcodes =====
    idx_series = adata.obs_names.to_series()
    rc = idx_series.str.extract(r"s_002um_(\d+)_(\d+)-1")
    if rc.isnull().any().any():
        print("Warning: Failed to parse some barcodes! Please check the format.")

    rows = rc[0].astype(int).to_numpy()
    cols = rc[1].astype(int).to_numpy()

    # ===== 2. MSF mask and seed =====
    msf_labels = adata.obs[mask_key].to_numpy()
    seed_labels = adata.obs[seed_key].to_numpy()

    # ===== 3. Offset starting coordinates to 0-based =====
    row_min, col_min = rows.min(), cols.min()
    rows0 = rows - row_min
    cols0 = cols - col_min

    H = rows0.max() + 1
    W = cols0.max() + 1
    print(f"Mask dimensions: H={H}, W={W}")

    # ===== 4. Initialize RGB mask image (black background) =====
    mask_img = np.zeros((H, W, 3), dtype=np.uint8)

    # ===== 5. Draw MSF mask (white) =====
    white_mask = (msf_labels != 0)
    mask_img[rows0[white_mask], cols0[white_mask]] = [255, 255, 255]

    # ===== 6. Draw Seed mask (red) =====
    seed_mask = (seed_labels > 0)
    mask_img[rows0[seed_mask], cols0[seed_mask]] = [255, 0, 0]

    # ===== 7. Display / Save =====
    plt.figure(figsize=figsize)
    plt.imshow(mask_img, origin="upper")
    plt.axis("off")
    plt.tight_layout()

    if output_path is not None:
        # <--- 2. New safety modification: Automatically create the image output directory to prevent FileNotFoundError
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=800)
        print(f"Mask saved to: {output_path}")

    plt.show()


def build_voronoi_from_seed(adata, seed_key="seed", use_barcode=True):
    """
    Construct discrete Voronoi based on adata.obs[seed_key] and bin coordinates (parsed from barcodes):
    - Each seed (>0) has a centroid
    - Each bin belongs to the nearest seed centroid
    Results are written to adata.obs["seed_voronoi"], representing the "theoretical territory".

    Returns:
        voronoi_owner: np.ndarray, length = n_obs, where each element corresponds to the seed_id (or 0 representing no seed)
    """

    # ===== 1. Parse coordinates (row, col) for each bin =====
    idx_series = adata.obs_names.to_series()

    # Assuming barcode format is similar to: s_002um_00014_00074-1
    rc = idx_series.str.extract(r"s_002um_(\d+)_(\d+)-1")
    if rc.isnull().any().any():
        print("Warning: Failed to parse some barcodes, please check regex pattern/format.")

    rows = rc[0].astype(int).to_numpy()
    cols = rc[1].astype(int).to_numpy()

    # Offset to start from 0 (for ease of visualization later)
    row_min, col_min = rows.min(), cols.min()
    rows0 = rows - row_min
    cols0 = cols - col_min

    # ===== 2. Read seed labels =====
    seeds = adata.obs[seed_key].to_numpy().astype(np.int64)
    # Treat <0 as unlabeled
    seeds[seeds < 0] = 0

    # Retain only bins with a nucleus (seed > 0)
    has_seed_mask = seeds > 0
    if has_seed_mask.sum() == 0:
        raise ValueError("No nuclear labels > 0 found in seed, cannot build Voronoi.")

    seed_ids = seeds[has_seed_mask]
    seed_rows = rows0[has_seed_mask]
    seed_cols = cols0[has_seed_mask]

    # ===== 3. Compute centroid for each seed =====
    df_seed = pd.DataFrame({
        "seed": seed_ids,
        "row": seed_rows,
        "col": seed_cols,
    })

    # Group by each seed and calculate the average coordinates as the centroid
    centroids = df_seed.groupby("seed")[["row", "col"]].mean()
    centroid_coords = centroids[["row", "col"]].to_numpy()  # (K, 2)
    centroid_seeds = centroids.index.to_numpy()  # Corresponding seed ID

    print(f"Total of {len(centroid_seeds)} seeds with centroid coordinates available for Voronoi.")

    # ===== 4. Perform nearest centroid assignment for all bins (discrete Voronoi) =====
    all_points = np.column_stack([rows0, cols0])  # (N, 2)
    tree = cKDTree(centroid_coords)
    dist, idx = tree.query(all_points)  # idx: Index of the nearest centroid for each point
    voronoi_owner = np.zeros(len(adata), dtype=np.int64)
    voronoi_owner[:] = 0  # Default 0 means "does not belong to any seed (theoretically won't happen)"
    voronoi_owner = centroid_seeds[idx]  # Fill with the seed ID corresponding to the nearest centroid

    # ===== 5. Write back to adata.obs =====
    adata.obs["seed_voronoi"] = voronoi_owner
    print("Successfully wrote Voronoi-assigned seeds to adata.obs['seed_voronoi'].")
    return adata


# ==========================================
# Core Data Loading and Processing Pipeline
# ==========================================

adata.var['gene'] = adata.var['gene'].astype(str).str.upper()
nnz_per_bin = np.diff(adata.X.indptr)

# Identify non-all-zero bins and apply the same preprocessing filtration
keep = nnz_per_bin > 0
adata = adata[keep].copy()
with open("vocab.json", "r") as f:
    vocab = json.load(f)
vocab_genes = {k for k in vocab.keys() if not k.startswith("<")}
genes = adata.var["gene"].astype(str)
keep_gene_mask = genes.isin(vocab_genes)
adata = adata[:, keep_gene_mask].copy()

X = adata.X
cell_sum = np.array(X.sum(axis=1)).ravel()
nonzero_cell_mask = cell_sum > 0
adata = adata[nonzero_cell_mask].copy()

summarize_seed_stats(adata, 'seed')
adata = build_voronoi_from_seed(adata, seed_key="seed")

edge_index = data['edge_index']  # shape (2, E)
alpha = data['alpha']  # shape (E,)
print("edge_index:", edge_index.shape)
print("alpha:", alpha.shape, alpha.min(), alpha.max())

edge_index = data["edge_index"].astype(np.int64)  # shape (2, E)
alpha = data["alpha"].astype(np.float32)  # shape (E, )

# —— If edge_index is 1-based, convert it to 0-based ——
if edge_index.min() == 1:
    edge_index -= 1

# ====== Node labels (nuclei segmentation seeds) ======
seeds = adata.obs["seed"].to_numpy().astype(np.int64)  # shape (N,)
seeds[seeds < 0] = 0
voronoi_owner = adata.obs["seed_voronoi"].to_numpy().astype(np.int64)

# Coordinates (use px/py to measure distance)
px = adata.obs["px"].to_numpy().astype(np.float32)
py = adata.obs["py"].to_numpy().astype(np.float32)

N = seeds.shape[0]
E = edge_index.shape[1]

print(f"N = {N}, E = {E}, seeds > 0 = {(seeds > 0).sum()}")

# ===== 1. Compute each nucleus centroid and 'nuclear radius' =====
nuc_mask = seeds > 0
df_nuc = pd.DataFrame({
    "seed": seeds[nuc_mask],
    "px": px[nuc_mask],
    "py": py[nuc_mask],
})

centroids = df_nuc.groupby("seed")[["px", "py"]].mean()

df_nuc = df_nuc.join(centroids, on="seed", rsuffix="_c")
dx = df_nuc["px"] - df_nuc["px_c"]
dy = df_nuc["py"] - df_nuc["py_c"]
df_nuc["dist"] = np.sqrt(dx * dx + dy * dy)

r_nuc = df_nuc.groupby("seed")["dist"].quantile(0.9)

max_seed_id = max(int(centroids.index.max()), int(voronoi_owner.max()))

center_px = np.zeros(max_seed_id + 1, dtype=np.float32)
center_py = np.zeros(max_seed_id + 1, dtype=np.float32)
radius_nuc = np.zeros(max_seed_id + 1, dtype=np.float32)

center_px[centroids.index.to_numpy()] = centroids["px"].to_numpy()
center_py[centroids.index.to_numpy()] = centroids["py"].to_numpy()
radius_nuc[r_nuc.index.to_numpy()] = r_nuc.to_numpy()

k_radius = 2.5
radius_limit = k_radius * radius_nuc

owner = voronoi_owner
cx_all = center_px[owner]
cy_all = center_py[owner]
dx_all = px - cx_all
dy_all = py - cy_all
dist_to_center = np.sqrt(dx_all * dx_all + dy_all * dy_all)

radius_limit_for_node = radius_limit[owner]

# ========= 2. Prepare Edges =========
u = edge_index[0]
v = edge_index[1]
valid_mask = (u >= 0) & (u < N) & (v >= 0) & (v < N)
u = u[valid_mask]
v = v[valid_mask]
alpha = alpha[valid_mask]

alpha_thresh = 0.5

strong_mask = alpha >= alpha_thresh
u = u[strong_mask]
v = v[strong_mask]
alpha = alpha[strong_mask]

E_valid = u.shape[0]
print(f"Edges after threshold (alpha >= {alpha_thresh}): {E_valid}")

same_voronoi = voronoi_owner[u] == voronoi_owner[v]
u = u[same_voronoi]
v = v[same_voronoi]
alpha = alpha[same_voronoi]
E_valid = u.shape[0]
print(f"Edges after Voronoi filtering: {E_valid}")

dist_ok = (dist_to_center[u] <= radius_limit_for_node[u]) & \
          (dist_to_center[v] <= radius_limit_for_node[v])

u = u[dist_ok]
v = v[dist_ok]
alpha = alpha[dist_ok]

E_valid = u.shape[0]
print(f"Edges after distance filtering (k_radius={k_radius}): {E_valid}")

cost = 1.0 - alpha


# ========= 3. Union-Find Structure =========
class UnionFind:
    def __init__(self, n, labels):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)
        self.comp_label = labels.copy()

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_seeded(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return

        lx = self.comp_label[px]
        ly = self.comp_label[py]

        if lx > 0 and ly > 0 and lx != ly:
            return

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        lx = self.comp_label[px]
        ly = self.comp_label[py]
        if lx == 0 and ly == 0:
            self.comp_label[px] = 0
        elif lx == 0 and ly > 0:
            self.comp_label[px] = ly
        elif lx > 0 and ly == 0:
            self.comp_label[px] = lx
        else:
            self.comp_label[px] = lx


uf = UnionFind(N, seeds)

# ========= 4. Sort Edges by Cost and Attempt Merge Sequentially =========
order = np.argsort(cost)
u_sorted = u[order]
v_sorted = v[order]

print("Start seeded MSF growth (this may take a while)...")

for i in tqdm(range(E_valid), desc="seed MSF growth"):
    uf.union_seeded(u_sorted[i], v_sorted[i])

print("MSF growth done.")

# ========= 5. Obtain Final Label for Each Node =========
final_labels = np.zeros(N, dtype=np.int64)
for i in range(N):
    root = uf.find(i)
    final_labels[i] = uf.comp_label[root]

print("Final labels stats:")
print("  total nodes:", N)
print("  unlabeled (0):", np.sum(final_labels == 0))
print("  unique labels (excluding 0):", len(np.unique(final_labels[final_labels > 0])))

# ========= 6. Write Back to adata and Save =========
adata.obs["msf"] = final_labels

summarize_seed_stats(adata, "msf")
visualize_msf_mask_with_seeds(adata, mask_key="msf", seed_key="seed", output_path=f"figure/{sample_id}/demo.pdf")

# ==================== 3. Core Modification: Safely align and write final_labels to adata2 based on Barcode index ====================
# Read the unfiltered, full-volume AnnData
adata2 = sc.read_h5ad(all_h5ad_path)
labels_series = pd.Series(final_labels, index=adata.obs_names)
adata2.obs["demo"] = labels_series.reindex(adata2.obs_names, fill_value=0).astype(np.int64)
print(adata2)
adata2.write_h5ad(all_h5ad_path)
