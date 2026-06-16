import numpy as np
import torch
import scanpy as sc
from scgpt.tasks.cell_emb import embed_data
from scipy.sparse import issparse, csr_matrix
import json
import os

# ==================== 1. Dynamically Patch AnnData View Classes ====================
try:
    from anndata._core.views import SparseCSRMatrixView, SparseCSCMatrixView
    SparseCSRMatrixView.A = property(lambda self: self.toarray())
    SparseCSCMatrixView.A = property(lambda self: self.toarray())
    print("Successfully patched SparseCSRMatrixView.A attribute")
except Exception as e:
    print(f"Failed to patch view attributes: {e}")

# ==================== 1.1 Dynamically Read Global Configuration ====================
config_path = "../pipeline_config.json"
sample_id = "mouse_brain"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sample_id = cfg.get("SAMPLE_ID", "mouse_brain")

file_fold = f"../{sample_id}_all.h5ad"

output_fold = f"{sample_id}_temp.npy"  # Dynamically named

# ==================== 2. Load Data and Standardize Matrix ====================
adata = sc.read_h5ad(file_fold)

if issparse(adata.X):
    adata.X = csr_matrix(adata.X)
else:
    adata.X = np.asarray(adata.X)

adata.var['gene'] = adata.var['gene'].astype(str).str.upper()

model_dir = "pretrained/scGPT_human"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 3. Load Vocabulary and Perform Cell Quality Filtering ====================
vocab_path = os.path.join(model_dir, "vocab.json")
with open(vocab_path, "r") as f:
    vocab = json.load(f)

# Find genes in adata that actually exist in the scGPT vocabulary
in_vocab_mask = adata.var['gene'].isin(vocab.keys())
matched_genes_count = in_vocab_mask.sum()
print(f"Number of matched genes in vocabulary: {matched_genes_count} / {adata.n_vars}")

if matched_genes_count == 0:
    raise ValueError("No genes matched the scGPT vocabulary. Please check the naming in adata.var['gene'].")

# Extract the subset containing only the matched genes
adata_matched = adata[:, in_vocab_mask].copy()

# Crucial: Filter out cells with a sum of expression equal to 0 among these matched genes
before_cells = adata_matched.n_obs
sc.pp.filter_cells(adata_matched, min_counts=1)  # Filter cells with count < 1
after_cells = adata_matched.n_obs
print(f"Filtered out {before_cells - after_cells} cells with zero expression across matched genes.")

# Assign the filtered safe data back to adata
adata = adata_matched

# ==================== 4. Run Embedding Extraction ====================
adata = embed_data(
    adata_or_file=adata,
    model_dir=model_dir,
    gene_col="gene",
    max_length=1200,
    batch_size=64,
    obs_to_save=None,
    device=device,
    use_fast_transformer=False,
    return_new_adata=False
)

cell_embeddings = adata.obsm["X_scGPT"]
print("Shape of cell embeddings:", cell_embeddings.shape)
#print(cell_embeddings)

np.save(output_fold, cell_embeddings)
print("Saving completed")