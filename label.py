import scanpy as sc
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

# Read h5ad file
config_path = "pipeline_config.json"
sample_id = "mouse_brain_demo"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sample_id = cfg.get("SAMPLE_ID", "mouse_brain_demo")

adata = sc.read_h5ad(f"{sample_id}_seed.h5ad")
print(adata.obs['seed'].nunique())

# 1. Extract barcode and seed
barcodes = adata.obs.index.astype(str).values  # index is barcode
seeds = adata.obs["seed"].to_numpy()          # int32, length = n_bins

n_bins = len(barcodes)
print("Total bins:", n_bins)

# 2. Parse (row, col) from barcode, e.g., s_002um_00014_00074-1 → 14, 74
rows = np.zeros(n_bins, dtype=int)
cols = np.zeros(n_bins, dtype=int)

# Write regular expression based on your barcode format
pattern = re.compile(r"s_002um_(\d+)_(\d+)-1")

for i, b in enumerate(barcodes):
    m = pattern.match(b)
    if m is None:
        raise ValueError(f"Unexpected barcode format: {b}")
    rows[i] = int(m.group(1))  # 00014 → 14
    cols[i] = int(m.group(2))  # 00074 → 74

print("row range:", rows.min(), rows.max())
print("col range:", cols.min(), cols.max())

# 3. Build a lookup table from (row, col) → index to easily locate 8-neighborhood
rc2idx = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}

# 4. Iterate over each bin, find the 8-neighborhood, and generate labeled edges based on seed
edges = []   # Each item: [row1, col1, row2, col2, label]

# Offset of the 8-neighborhood (excluding 0,0)
OFFS = [(-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)]

for i in tqdm(range(n_bins), desc="Building labeled edges"):
    seed_i = seeds[i]
    if seed_i == -1:
        # No corresponding nucleus, skip this bin directly
        continue

    r_i, c_i = rows[i], cols[i]

    for dr, dc in OFFS:
        r_j = r_i + dr
        c_j = c_i + dc
        key_j = (r_j, c_j)

        # Check if neighbor exists
        j = rc2idx.get(key_j, None)
        if j is None:
            continue

        seed_j = seeds[j]
        if seed_j == -1:
            # Neighbor also has no nucleus, ignore this edge
            continue

        # To avoid duplicates of (i,j) and (j,i), keep only the case where i < j
        if i >= j:
            continue

        # Same seed → 1, different seed → -1
        if seed_i == seed_j:
            label = 1
        else:
            label = -1

        edges.append([r_i, c_i, r_j, c_j, label])

# 5. Save as DataFrame (more intuitive and convenient)
edge_df = pd.DataFrame(edges, columns=["row1", "col1", "row2", "col2", "label"])

print("Total edges with labels:", len(edge_df))
print(edge_df.head())
print("Positive labels (1):", (edge_df["label"] == 1).sum())
print("Negative labels (-1):", (edge_df["label"] == -1).sum())

# 6. Save statistics to a JSON file
label_stats = {
    "pos_samples": int((edge_df["label"] == 1).sum()),
    "sub_samples": int((edge_df["label"] == -1).sum()), # Convert to int type supported by JSON
}
with open(f"label_stats_{sample_id}.json", "w", encoding="utf-8") as f:
    json.dump(label_stats, f, indent=4)

edge_df.to_csv(f"bin_edge_labels_{sample_id}.csv", index=False)
