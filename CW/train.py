import re
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from model import EndToEndModel
import json
from tqdm import tqdm
import os
import gc
import scipy.sparse as sp
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# 0. Configuration: Enable GPU
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Limit CPU threads, leaving headroom for data loading
torch.set_num_threads(8)

# ==================== Dynamically Read Global Configuration ====================
config_path = "pipeline_config.json"
sample_id = "mouse_brain_demo"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sample_id = cfg.get("SAMPLE_ID", "mouse_brain_demo")
    IMG_BARCODE_PATH = f"UNI/final_barcodes_{sample_id}.npy"
    IMG_MATRIX_PATH = f"UNI/final_matrix_{sample_id}.npy"
    emb_path = f"scGPT/{sample_id}_temp.npy"

# ==========================================
# 1. Data Preparation
# ==========================================
print("Loading h5ad data...")
adata = sc.read_h5ad(f"{sample_id}_seed.h5ad")
adata.var['gene'] = adata.var['gene'].astype(str).str.upper()
nnz_per_bin = np.diff(adata.X.indptr)
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

# ==========================================
# 2. Load Embeddings
# ==========================================

node_emb = np.load(emb_path).astype(np.float32)
N_gene, d_node = node_emb.shape

# [Safety Fix]: Automatically detect and clean NaNs in scGPT
if np.isnan(node_emb).any():
    print("Warning: Detected NaN values in scGPT gene embeddings! Automatically filled with 0...")
    node_emb = np.nan_to_num(node_emb, nan=0.0, posinf=0.0, neginf=0.0)

print("Loading raw image matrix...")
try:
    img_barcodes_raw = np.load(IMG_BARCODE_PATH, allow_pickle=True)
except FileNotFoundError:
    print("change")

n_img_samples = len(img_barcodes_raw)

print(f"Reading {n_img_samples}x1024 matrix into RAM...")
img_matrix_mmap = np.memmap(IMG_MATRIX_PATH, dtype='float32', mode='r', shape=(n_img_samples, 1024))
img_matrix_raw = np.array(img_matrix_mmap)
del img_matrix_mmap

# [Safety Fix]: Automatically detect and clean NaNs in UNI
if np.isnan(img_matrix_raw).any():
    print("Warning: Detected NaN values in UNI image matrix! Automatically filled with 0...")
    img_matrix_raw = np.nan_to_num(img_matrix_raw, nan=0.0, posinf=0.0, neginf=0.0)

img_bc_to_idx = {bc: i for i, bc in enumerate(img_barcodes_raw)}

print("Aligning data...")
target_barcodes = adata.obs_names.to_numpy()
aligned_img_emb = np.zeros((N_gene, 1024), dtype=np.float32)
for i, bc in tqdm(enumerate(target_barcodes), total=N_gene, desc="Aligning"):
    if bc in img_bc_to_idx:
        aligned_img_emb[i] = img_matrix_raw[img_bc_to_idx[bc]]

# Convert to Tensor and temporarily cache on CPU
x_gene_cpu = torch.from_numpy(node_emb).float()
x_img_cpu = torch.from_numpy(aligned_img_emb).float()

del node_emb, aligned_img_emb, img_matrix_raw
gc.collect()

# ==========================================
# 3. Build Full Graph & Scipy Sparse Matrix
# ==========================================
print("Building Graph...")
barcodes = adata.obs.index.astype(str).values
rows = np.zeros(N_gene, dtype=int)
cols = np.zeros(N_gene, dtype=int)
pattern = re.compile(r"s_002um_(\d+)_(\d+)-1")
for i, b in enumerate(barcodes):
    m = pattern.match(b)
    if m: rows[i], cols[i] = int(m.group(1)), int(m.group(2))

rc2idx = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

edges = []
for i in tqdm(range(N_gene), desc="Generating edges"):
    r_i, c_i = rows[i], cols[i]
    for dr, dc in OFFS:
        key_j = (r_i + dr, c_i + dc)
        j = rc2idx.get(key_j)
        if j is not None and i != j:
            edges.append((min(i, j), max(i, j)))

edges = np.unique(np.array(edges, dtype=np.int64), axis=0)
src = edges[:, 0]
dst = edges[:, 1]
E = len(src)
print(f"Total edges: {E}")

print("Building Scipy CSR Matrix for sampling...")
vals = np.ones(E, dtype=bool)
full_src = np.concatenate([src, dst])
full_dst = np.concatenate([dst, src])
full_vals = np.concatenate([vals, vals])

adj_csr = sp.csr_matrix((full_vals, (full_src, full_dst)), shape=(N_gene, N_gene))
print("CSR Matrix built.")

# ==========================================
# 4. Prepare Labels and Custom Loader
# ==========================================
print("Preparing Labels...")
sample_id = "mouse_brain_demo"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sample_id = cfg.get("SAMPLE_ID", "mouse_brain_demo")
edge_df = pd.read_csv(f"bin_edge_labels_{sample_id}_cut12.csv")

labeled_edges = []
for _, r in tqdm(edge_df.iterrows(), total=len(edge_df), desc="Parsing labels"):
    r1, c1, r2, c2 = int(r["row1"]), int(r["col1"]), int(r["row2"]), int(r["col2"])
    lbl = 1.0 if int(r["label"]) == 1 else 0.0

    i, j = rc2idx.get((r1, c1)), rc2idx.get((r2, c2))
    if i is not None and j is not None:
        labeled_edges.append((min(i, j), max(i, j), lbl))

labeled_edges = np.array(labeled_edges, dtype=object)
num_labeled = len(labeled_edges)
print(f"Labeled edges: {num_labeled}")


def get_batch_subgraph(batch_indices, hop=1):
    """
    Input: A batch of labeled edge indices
    Output: Batch Data (moved to GPU)
    """
    batch_data = labeled_edges[batch_indices]
    u_seeds = batch_data[:, 0].astype(np.int64)
    v_seeds = batch_data[:, 1].astype(np.int64)
    labels = batch_data[:, 2].astype(np.float32)

    seed_nodes = np.unique(np.concatenate([u_seeds, v_seeds]))

    if hop > 0:
        neighbors_mat = adj_csr[seed_nodes]
        neighbor_nodes = neighbors_mat.indices
        all_nodes = np.unique(np.concatenate([seed_nodes, neighbor_nodes]))
    else:
        all_nodes = seed_nodes

    sub_adj = adj_csr[all_nodes, :][:, all_nodes]
    sub_coo = sub_adj.tocoo()

    row_local = torch.from_numpy(sub_coo.row).long()
    col_local = torch.from_numpy(sub_coo.col).long()
    batch_edge_index = torch.stack([row_local, col_local], dim=0)

    batch_x_gene = x_gene_cpu[all_nodes]
    batch_x_img = x_img_cpu[all_nodes]

    u_local = np.searchsorted(all_nodes, u_seeds)
    v_local = np.searchsorted(all_nodes, v_seeds)

    batch_label_u = torch.from_numpy(u_local).long()
    batch_label_v = torch.from_numpy(v_local).long()
    batch_labels = torch.from_numpy(labels).float()

    return (batch_x_gene, batch_x_img, batch_edge_index,
            batch_label_u, batch_label_v, batch_labels)


# ==========================================
# 6. Training Loop (Manual Batching)
# ==========================================
model = EndToEndModel(gene_dim=512, img_dim=1024, gnn_hidden=64, gnn_layers=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

print("\nStarting GPU Custom-Batch Training (No torch-sparse needed)...")

BATCH_SIZE = 4096
num_batches = (num_labeled + BATCH_SIZE - 1) // BATCH_SIZE
indices = np.arange(num_labeled)

# ===== Early Stopping Settings =====
best_acc = -1.0
no_improve_epochs = 0
patience = 30

# ===== For recording training curves =====
epoch_list = []
loss_list = []
acc_list = []

os.makedirs("weight", exist_ok=True)
os.makedirs("curve", exist_ok=True)

for epoch in range(1, 300):
    start_time = time.time()
    model.train()
    np.random.shuffle(indices)

    total_loss = 0
    total_acc = 0

    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}", leave=False)

    for i in pbar:
        # 1. Prepare Batch Data
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, num_labeled)
        batch_idx = indices[start_idx:end_idx]

        (b_x_g, b_x_i, b_edge_index,
         b_u, b_v, b_y) = get_batch_subgraph(batch_idx, hop=1)

        # 2. Move to GPU
        b_x_g = b_x_g.to(device)
        b_x_i = b_x_i.to(device)
        b_edge_index = b_edge_index.to(device)
        b_u = b_u.to(device)
        b_v = b_v.to(device)
        b_y = b_y.to(device)

        # 3. Training
        optimizer.zero_grad()

        # ==================== Core Modification: Directly call MoE to bypass redundant GNN computation graphs ====================
        # This step only runs the MoE module to obtain the fused 512-dimensional features x_fused (completely resolving backward errors and accelerating computation)
        x_fused, gates = model.moe(b_x_g, b_x_i)
        # ===================================================================================

        # ==================== Manually update GNN nodes to obtain 64-dimensional features ====================
        # Explicitly invoke convs to safely and sequentially calculate the updated 64-dimensional features on the GPU
        node_emb_final = x_fused
        for conv in model.gnn.convs:
            node_emb_final = conv(node_emb_final, b_edge_index)
            node_emb_final = F.relu(node_emb_final)
        # At this point, node_emb_final shape is strictly [N_local, 64]
        # ===================================================================================

        # ==================== Directly connect predictions to prevent NaN ====================
        # Extract the 64-dimensional representation using local node indices at both ends of the labeled edges
        h_u = node_emb_final[b_u]  # Shape: [B, 64]
        h_v = node_emb_final[b_v]  # Shape: [B, 64]
        h_abs = torch.abs(h_u - h_v)

        # Concatenate endpoint representations; the resulting dimension is 192 (64 * 3)
        h_edge_labeled = torch.cat([h_u, h_v, h_abs], dim=-1)  # Shape: [B, 192]

        # Feed into classifier
        logit_labeled = model.gnn.edge_mlp(h_edge_labeled).squeeze(-1)  # Shape: [B]
        # ===================================================================

        loss = criterion(logit_labeled, b_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        with torch.no_grad():
            pred = (torch.sigmoid(logit_labeled) > 0.5).float()
            acc = (pred == b_y).float().mean().item()
            total_acc += acc

        pbar.set_postfix({"Loss": loss.item(), "Acc": acc})

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    epoch_list.append(epoch)
    loss_list.append(avg_loss)
    acc_list.append(avg_acc)

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f} | Time: {elapsed:.2f}s")

    # ===== Early Stopping =====
    if avg_acc > best_acc:
        best_acc = avg_acc
        no_improve_epochs = 0
        torch.save(model.state_dict(), "weight/best.pt")
        print(f"  New best Avg Acc: {best_acc:.4f}, model saved to weight/best.pt")
    else:
        no_improve_epochs += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ⚠️ No improvement ({no_improve_epochs}/{patience})")

        if no_improve_epochs >= patience:
            print(f"\n Early stopping triggered at epoch {epoch}")
            print(f"Best Avg Acc: {best_acc:.4f}")
            break

# ==========================================
# 7. Save Results (Inference logic remains unchanged)
# ==========================================
print("Training finished. Running full inference...")
best_path = "weight/best.pt"
if os.path.exists(best_path):
    model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()
model.cpu()
torch.cuda.empty_cache()

full_edge_index = torch.from_numpy(edges.T).long()

print(f"Inference on {N_gene} nodes and {len(edges)} edges...")

def get_infer_subgraph(target_nodes_np, hop=1, x=None):
    seed_nodes = np.unique(target_nodes_np)
    if hop > 0:
        neighbors_mat = adj_csr[seed_nodes]
        neighbor_nodes = neighbors_mat.indices
        all_nodes = np.unique(np.concatenate([seed_nodes, neighbor_nodes]))
    else:
        all_nodes = seed_nodes

    sub_adj = adj_csr[all_nodes, :][:, all_nodes]
    sub_coo = sub_adj.tocoo()

    row_local = torch.from_numpy(sub_coo.row).long()
    col_local = torch.from_numpy(sub_coo.col).long()
    sub_edge_index = torch.stack([row_local, col_local], dim=0)

    sub_x = x[all_nodes]
    target_mask_np = np.isin(all_nodes, seed_nodes)
    target_mask = torch.from_numpy(target_mask_np)

    return sub_x, sub_edge_index, all_nodes, target_mask

with torch.no_grad():
    N = x_gene_cpu.shape[0]
    fused_dim = 512

    node_chunk = 300000
    print(f"Computing MoE fusion in chunks...")

    node_emb_fused = torch.empty((N, fused_dim), dtype=torch.float32, device="cpu")
    gates = torch.empty((N, 2), dtype=torch.float32, device="cpu")

    for start in tqdm(range(0, N, node_chunk), desc="MoE node chunks"):
        end = min(start + node_chunk, N)
        xg = x_gene_cpu[start:end]
        xi = x_img_cpu[start:end]
        fused_chunk, gates_chunk = model.moe(xg, xi)
        node_emb_fused[start:end] = fused_chunk
        gates[start:end] = gates_chunk

    print("3.2")
    h_in = node_emb_fused
    N = h_in.size(0)
    hidden_dim = model.gnn.convs[-1].out_channels
    h_out = torch.empty((N, hidden_dim), dtype=torch.float32, device="cpu")

    node_chunk = 50000
    for start in tqdm(range(0, N, node_chunk), desc="GNN node chunks"):
        end = min(start + node_chunk, N)
        target_nodes_np = np.arange(start, end, dtype=np.int64)
        sub_x, sub_edge_index, all_nodes, target_mask = get_infer_subgraph(
            target_nodes_np, hop=1, x=h_in
        )
        sub_h = sub_x
        for conv in model.gnn.convs:
            sub_h = conv(sub_h, sub_edge_index)
            sub_h = F.relu(sub_h)
        h_out[start:end] = sub_h[target_mask]
        del sub_x, sub_edge_index, sub_h, all_nodes, target_mask
        gc.collect()
    h = h_out

    edge_index = full_edge_index
    src, dst = edge_index
    E_total = src.shape[0]

    chunk_size = 500000
    logit_list = []

    print(f"Computing edge probabilities in chunks...")
    for start in tqdm(range(0, E_total, chunk_size), desc="Edge chunks"):
        end = min(start + chunk_size, E_total)
        src_chunk = src[start:end]
        dst_chunk = dst[start:end]
        h_src = h[src_chunk]
        h_dst = h[dst_chunk]
        h_abs = torch.abs(h_src - h_dst)
        h_edge = torch.cat([h_src, h_dst, h_abs], dim=-1)
        logit_chunk = model.gnn.edge_mlp(h_edge).squeeze(-1)
        logit_list.append(logit_chunk.cpu())

    logit_all = torch.cat(logit_list, dim=0)
    prob_all = torch.sigmoid(logit_all)

alpha = prob_all.numpy()
edge_index_np = edges.T

os.makedirs(f"weight/{sample_id}", exist_ok=True)
np.savez(
    f"weight/{sample_id}/demo.npz",
    edge_index=edge_index_np,
    alpha=alpha,
    rows=rows,
    cols=cols
)
print("Edge probabilities saved.")

os.makedirs("weight/MoE", exist_ok=True)
gates_np = gates.numpy()
np.save(f"weight/MoE/moe_weights_{sample_id}_.npy", gates_np)
print(f"MoE weights saved. Shape: {gates_np.shape}")
print("Done!")

df_curve = pd.DataFrame({
    "epoch": epoch_list,
    "loss": loss_list,
    "accuracy": acc_list
})
df_curve.to_csv(f"curve/training_{sample_id}.csv", index=False)
print("Saved training_curve.csv")

plt.figure(figsize=(8, 5))
plt.plot(epoch_list, loss_list, label="Loss", color="tab:red")
plt.xlabel("Epoch")
plt.ylabel("Loss", color="tab:red")
plt.tick_params(axis="y", labelcolor="tab:red")

ax2 = plt.gca().twinx()
ax2.plot(epoch_list, acc_list, label="Accuracy", color="tab:blue")
ax2.set_ylabel("Accuracy", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("Training Loss & Accuracy over Epochs")
plt.grid(alpha=0.3)
plt.tight_layout()
out_png = f"curve/training_{sample_id}.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
#plt.show()