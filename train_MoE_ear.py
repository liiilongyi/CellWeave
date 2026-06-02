import re
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from model import EndToEndModel
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
# 0. 配置：开启 GPU
# ==========================================
# 既然我们手动切分数据，就可以放心使用 GPU 了
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# 限制 CPU 线程，留给数据加载
torch.set_num_threads(8)

#IMG_MATRIX_PATH = '<IMAGE_EMBEDDING_MATRIX_NPY>'
#IMG_BARCODE_PATH = '<IMAGE_EMBEDDING_BARCODES_NPY>'

IMG_BARCODE_PATH = '<IMAGE_EMBEDDING_BARCODES_NPY>'
IMG_MATRIX_PATH = '<IMAGE_EMBEDDING_MATRIX_NPY>'

# ==========================================
# 1. 数据准备 (保持不变)
# ==========================================
print("Loading h5ad data...")
#adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
#adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
adata.var['gene'] = adata.var['gene'].astype(str).str.upper()
nnz_per_bin = np.diff(adata.X.indptr)
keep = nnz_per_bin > 0
adata = adata[keep].copy()

with open("<SCGPT_VOCAB_JSON>", "r") as f:
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
# 2. 加载 Embedding (Raw Binary)
# ==========================================
print("Loading gene embeddings...")
#emb_path = "<EMBEDDING_NPY>"
emb_path = "<EMBEDDING_NPY>"



node_emb = np.load(emb_path).astype(np.float32) # 512
N_gene, d_node = node_emb.shape

print("Loading raw image matrix...")
try:
    img_barcodes_raw = np.load(IMG_BARCODE_PATH, allow_pickle=True)
except FileNotFoundError:
    #IMG_BARCODE_PATH = '<IMAGE_EMBEDDING_BARCODES_NPY>'
    print("change")
    #img_barcodes_raw = np.load(IMG_BARCODE_PATH, allow_pickle=True)
n_img_samples = len(img_barcodes_raw)

print(f"Reading {n_img_samples}x1024 matrix into RAM...")
img_matrix_mmap = np.memmap(IMG_MATRIX_PATH, dtype='float32', mode='r', shape=(n_img_samples, 1024))
img_matrix_raw = np.array(img_matrix_mmap)

del img_matrix_mmap

img_bc_to_idx = {bc: i for i, bc in enumerate(img_barcodes_raw)}

print("Aligning data...")
target_barcodes = adata.obs_names.to_numpy()
aligned_img_emb = np.zeros((N_gene, 1024), dtype=np.float32)
for i, bc in tqdm(enumerate(target_barcodes), total=N_gene, desc="Aligning"):
    if bc in img_bc_to_idx:
        aligned_img_emb[i] = img_matrix_raw[img_bc_to_idx[bc]]

# 转 Tensor，依然放在 CPU 上，训练时再搬运
x_gene_cpu = torch.from_numpy(node_emb).float()
x_img_cpu = torch.from_numpy(aligned_img_emb).float()

del node_emb, aligned_img_emb, img_matrix_raw
gc.collect()

# ==========================================
# 3. 构建全图 & Scipy 稀疏矩阵
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
# 这里的 edge_index 用于构建 CSR，不再直接转 Tensor
src = edges[:, 0]
dst = edges[:, 1]
E = len(src)
print(f"Total edges: {E}")

# 【核心】：构建 Scipy CSR 矩阵，用于快速查找邻居
# 这是一个非常高效的 CPU 图结构，不依赖 pytorch-sparse
print("Building Scipy CSR Matrix for sampling...")
# data 全是 1，我们只关心结构
vals = np.ones(E, dtype=bool)
# 构建无向图的邻接矩阵 (对称)
# 为了方便，我们把 (u,v) 和 (v,u) 都放进去
full_src = np.concatenate([src, dst])
full_dst = np.concatenate([dst, src])
full_vals = np.concatenate([vals, vals])

adj_csr = sp.csr_matrix((full_vals, (full_src, full_dst)), shape=(N_gene, N_gene))
print("CSR Matrix built.")

# ==========================================
# 4. 准备标签和自定义 Loader
# ==========================================
print("Preparing Labels...")
#edge_df = pd.read_csv("<EDGE_LABELS_CSV>")
#edge_df = pd.read_csv("<EDGE_LABELS_CSV>")
edge_df = pd.read_csv("<EDGE_LABELS_CSV>")
#edge_df = pd.read_csv("<EDGE_LABELS_CSV>")

# 提取有标签的边 (u, v, label)
labeled_edges = []
for _, r in tqdm(edge_df.iterrows(), total=len(edge_df), desc="Parsing labels"):
    r1, c1, r2, c2 = int(r["row1"]), int(r["col1"]), int(r["row2"]), int(r["col2"])
    lbl = 1.0 if int(r["label"]) == 1 else 0.0

    i, j = rc2idx.get((r1, c1)), rc2idx.get((r2, c2))
    if i is not None and j is not None:
        labeled_edges.append((min(i, j), max(i, j), lbl))

labeled_edges = np.array(labeled_edges, dtype=object)  # (N_labels, 3)
num_labeled = len(labeled_edges)
print(f"Labeled edges: {num_labeled}")


# 【自定义采样器】不依赖 pyg-lib
def get_batch_subgraph(batch_indices, hop=1):
    """
    输入：一批有标签边的索引
    输出：Batch Data (放入 GPU)
    逻辑：
    1. 取出这批边的节点 (Seed Nodes)
    2. 在 CSR 矩阵中找这些节点的 1-hop 邻居
    3. 构建子图 (Induced Subgraph)
    """
    # 1. 获取种子节点
    batch_data = labeled_edges[batch_indices]
    u_seeds = batch_data[:, 0].astype(np.int64)
    v_seeds = batch_data[:, 1].astype(np.int64)
    labels = batch_data[:, 2].astype(np.float32)

    seed_nodes = np.unique(np.concatenate([u_seeds, v_seeds]))

    # 2. 采样邻居 (1-hop)
    # 使用 CSR 矩阵切片，非常快
    if hop > 0:
        # adj_csr[nodes] 返回的是这些节点的邻居行
        neighbors_mat = adj_csr[seed_nodes]
        # 获取非零元素的列索引，即邻居 ID
        neighbor_nodes = neighbors_mat.indices
        # 合并所有节点
        all_nodes = np.unique(np.concatenate([seed_nodes, neighbor_nodes]))
    else:
        all_nodes = seed_nodes

    # 3. 构建子图的 edge_index
    # 这一步：提取 all_nodes 内部的所有边 (Induced Subgraph)
    # 方法：sub_adj = adj[all_nodes][:, all_nodes]
    # 但我们需要重新编号：全局 ID -> 局部 ID (0 ~ len(all_nodes)-1)

    # 映射表: global_id -> local_id
    # 使用 np.searchsorted 来快速映射 (前提是 all_nodes 有序，np.unique 保证了有序)
    # 但 scipy 切片更方便

    sub_adj = adj_csr[all_nodes, :][:, all_nodes]
    sub_coo = sub_adj.tocoo()

    row_local = torch.from_numpy(sub_coo.row).long()
    col_local = torch.from_numpy(sub_coo.col).long()
    batch_edge_index = torch.stack([row_local, col_local], dim=0)

    # 4. 准备特征
    # 此时 all_nodes 是全局索引
    batch_x_gene = x_gene_cpu[all_nodes]
    batch_x_img = x_img_cpu[all_nodes]

    # 5. 准备 Label 对应的边在子图中的索引
    # 我们需要把 u_seeds, v_seeds 映射为 local id
    # 利用 searchsorted (all_nodes 已经排序)
    u_local = np.searchsorted(all_nodes, u_seeds)
    v_local = np.searchsorted(all_nodes, v_seeds)

    # 转换为 Tensor
    batch_label_u = torch.from_numpy(u_local).long()
    batch_label_v = torch.from_numpy(v_local).long()
    batch_labels = torch.from_numpy(labels).float()

    return (batch_x_gene, batch_x_img, batch_edge_index,
            batch_label_u, batch_label_v, batch_labels)


# ==========================================
# 6. 训练循环 (手动 Batch)
# ==========================================
model = EndToEndModel(gene_dim=512, img_dim=1024, gnn_hidden=64, gnn_layers=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()


print("\nStarting GPU Custom-Batch Training (No torch-sparse needed)...")

BATCH_SIZE = 4096
num_batches = (num_labeled + BATCH_SIZE - 1) // BATCH_SIZE
indices = np.arange(num_labeled)

# ===== Early Stopping 设置 =====
best_acc = -1.0          # 历史最优 avg_acc
no_improve_epochs = 0   # 连续未提升计数
patience = 30            # 允许连续不提升的 epoch 数

# ===== 用于记录训练曲线 =====
epoch_list = []
loss_list = []
acc_list = []

for epoch in range(1, 300):
    start_time = time.time()
    model.train()
    np.random.shuffle(indices)  # 每个 epoch 打乱顺序

    total_loss = 0
    total_acc = 0

    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}", leave=False)

    for i in pbar:
        # 1. 准备 Batch 数据 (CPU上运行，scipy很快)
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, num_labeled)
        batch_idx = indices[start_idx:end_idx]

        # 调用我们手写的采样器
        (b_x_g, b_x_i, b_edge_index,
         b_u, b_v, b_y) = get_batch_subgraph(batch_idx, hop=1)

        # 2. 搬运到 GPU
        b_x_g = b_x_g.to(device)
        b_x_i = b_x_i.to(device)
        b_edge_index = b_edge_index.to(device)
        b_y = b_y.to(device)

        # 3. 训练
        optimizer.zero_grad()

        # 前向传播：计算子图中所有节点的特征和所有边的 logit
        # logit_all 对应 b_edge_index 中的每一条边
        logit_all, _, gates, _ = model(b_x_g, b_x_i, b_edge_index)

        # 4. 匹配 Label
        # 难点：我们需要从 logit_all (所有子图边) 中找到 b_u, b_v 对应的边
        # logit_all 的顺序与 b_edge_index 一致

        # 我们可以构建一个临时的 tensor 映射
        # 技巧：给边编码 id = u * N_local + v
        N_local = b_x_g.size(0)

        # 所有的消息传递边 ID
        all_edge_ids = b_edge_index[0] * N_local + b_edge_index[1]

        # 我们的目标标签边 ID
        target_edge_ids = b_u.to(device) * N_local + b_v.to(device)

        # 查找 target 在 all 中的位置
        # 由于 b_edge_index 是从 CSR 来的，通常是有序的，但不能保证
        # 使用 torch.isin 或者 广播查找太慢
        # 由于我们是在训练，我们可以假设 EdgeGNN 的输出包含了这些边
        # 实际上，get_batch_subgraph 中的 induced subgraph 包含了所有内部连接
        # 所以 b_u, b_v 这一对一定在 b_edge_index 中

        # 为了极速匹配，我们可以利用 Python 字典 (GPU tensor -> cpu list -> map -> index)
        # 或者利用 searchsorted (如果 edge_index 是排序的)
        # scipy 的 tocoo 不保证排序。

        # === 简单且健壮的方法 ===
        # 我们不在 EdgeGNN 输出里找。
        # 我们直接利用 GNN 更新后的 node embedding，手动算一下这几条边的 logit
        # 这要求我们访问 node_emb_fused

        _, _, _, node_emb_final = model(b_x_g, b_x_i, b_edge_index)

        # 手动计算 MLP(u, v)
        # 这需要知道 EdgeGNN 最后的分类头是什么。
        # 如果 model.py 里 EdgeGNN.decoder 是公开的，我们可以用。
        # 假设 model.gnn 内部最后一步是 self.decoder(cat(u, v)) 或 similar

        # *临时代替方案*：
        # 如果你无法修改 model.py，我们必须在 b_edge_index 找。
        # 让我们用一个 CPU map 快速做。

        msg_edges_cpu = b_edge_index.T.cpu().numpy()
        edge_map = {(u, v): k for k, (u, v) in enumerate(msg_edges_cpu)}

        # 找到索引
        target_indices = []
        valid_mask = []

        # b_u, b_v 已经是 local id
        u_cpu = b_u.cpu().numpy()
        v_cpu = b_v.cpu().numpy()

        for k in range(len(u_cpu)):
            key = (u_cpu[k], v_cpu[k])
            if key in edge_map:
                target_indices.append(edge_map[key])
                valid_mask.append(True)
            else:
                # 理论上 induced subgraph 一定包含原边，除非是有向图问题
                # 无向图 CSR 构造时我们是对称的，所以应该包含
                valid_mask.append(False)

        target_indices = torch.tensor(target_indices, device=device, dtype=torch.long)
        b_y_valid = b_y[valid_mask]

        logit_labeled = logit_all[target_indices]

        loss = criterion(logit_labeled, b_y_valid)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Acc
        with torch.no_grad():
            pred = (torch.sigmoid(logit_labeled) > 0.5).float()
            acc = (pred == b_y_valid).float().mean().item()
            total_acc += acc

        pbar.set_postfix({"Loss": loss.item(), "Acc": acc})

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    # ===== 记录当前 epoch 指标 =====
    epoch_list.append(epoch)
    loss_list.append(avg_loss)
    acc_list.append(avg_acc)

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f} | Time: {elapsed:.2f}s")

    # ===== Early Stopping 每个 epoch 都判断 =====
    if avg_acc > best_acc:
        best_acc = avg_acc
        no_improve_epochs = 0
        torch.save(model.state_dict(), "<MODEL_WEIGHTS_PT>")
        print(f"<MODEL_WEIGHTS_PT>")
    else:
        no_improve_epochs += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ⚠️ No improvement ({no_improve_epochs}/{patience})")

        if no_improve_epochs >= patience:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
            print(f"Best Avg Acc: {best_acc:.4f}")
            break

# ==========================================
# 7. 保存结果 (分块推理 + 保存 MoE 权重)
# ==========================================
print("Training finished. Running full inference to generate edge probabilities and save MoE weights...")
best_path = "<MODEL_WEIGHTS_PT>"
if os.path.exists(best_path):
    model.load_state_dict(torch.load(best_path, map_location=device))
    #print(f"Loaded best model from {best_path} (best_acc={best_acc:.4f})")
else:
    print("⚠️ <MODEL_WEIGHTS_PT> not found, using current model weights for inference.")
# 1. 将模型移回 CPU，清理 GPU 显存
model.eval()
model.cpu()
torch.cuda.empty_cache()

# 2. 准备全量数据
# x_gene_cpu 和 x_img_cpu 已经在 CPU 上
# 我们需要构建全量的 edge_index Tensor
# 注意：之前的 edges 变量是 (E, 2) 的 numpy 数组
full_edge_index = torch.from_numpy(edges.T).long()  # Shape: [2, E_total]

print(f"Inference on {N_gene} nodes and {len(edges)} edges...")

def get_infer_subgraph(target_nodes_np, hop=1, x=None):
    """
    target_nodes_np: np.ndarray, shape [B]，全局节点 id
    hop: 目前只实现 1-hop（和你训练一致）
    x: 全图节点特征 torch.Tensor [N, dim] (CPU)

    return:
      sub_x: torch.Tensor [N_sub, dim]
      sub_edge_index: torch.LongTensor [2, E_sub]  (局部编号)
      all_nodes: np.ndarray [N_sub] (全局 id，已排序)
      target_mask: torch.BoolTensor [N_sub] (哪些是 target nodes)
    """
    # 1) seed nodes
    seed_nodes = np.unique(target_nodes_np)

    # 2) 1-hop neighbors
    if hop > 0:
        neighbors_mat = adj_csr[seed_nodes]
        neighbor_nodes = neighbors_mat.indices
        all_nodes = np.unique(np.concatenate([seed_nodes, neighbor_nodes]))
    else:
        all_nodes = seed_nodes

    # 3) induced subgraph adjacency -> COO -> edge_index (local)
    sub_adj = adj_csr[all_nodes, :][:, all_nodes]
    sub_coo = sub_adj.tocoo()

    row_local = torch.from_numpy(sub_coo.row).long()
    col_local = torch.from_numpy(sub_coo.col).long()
    sub_edge_index = torch.stack([row_local, col_local], dim=0)

    # 4) features
    # x 是 CPU tensor，可以直接索引
    sub_x = x[all_nodes]

    # 5) target_mask: 在 all_nodes 里哪些属于 seed_nodes
    # 因为 all_nodes 和 seed_nodes 都是排序的 unique，可用 isin
    target_mask_np = np.isin(all_nodes, seed_nodes)
    target_mask = torch.from_numpy(target_mask_np)

    return sub_x, sub_edge_index, all_nodes, target_mask

with torch.no_grad():
    # ===== 3.1 先做 MoE 融合，得到节点输入特征 =====
    # x_gene_cpu, x_img_cpu 已经在 CPU 上
    # 【更改点 1】：捕获第二个返回值 gates (即权重)
    # ===== 3.1 先做 MoE 融合（分块），得到节点输入特征 + gates =====
    N = x_gene_cpu.shape[0]
    # 你原注释说 fused 是 [N, 512]，这里就直接用 512；如果不是 512，改成实际值
    fused_dim = 512

    node_chunk = 300000  # 节点分块大小：10万~50万都可，内存不够就调小
    print(f"Computing MoE fusion in chunks (node_chunk={node_chunk})...")

    # 预分配，避免 list 叠加再 cat 又吃一遍内存
    node_emb_fused = torch.empty((N, fused_dim), dtype=torch.float32, device="cpu")
    gates = torch.empty((N, 2), dtype=torch.float32, device="cpu")

    for start in tqdm(range(0, N, node_chunk), desc="MoE node chunks"):
        end = min(start + node_chunk, N)

        xg = x_gene_cpu[start:end]  # [chunk, d_gene]
        xi = x_img_cpu[start:end]  # [chunk, d_img]

        fused_chunk, gates_chunk = model.moe(xg, xi)  # 这里 cat 发生在 chunk 内，不会爆内存

        node_emb_fused[start:end] = fused_chunk
        gates[start:end] = gates_chunk

    print("3.2")
    # ===== 3.2 全图 GNN 节点更新（只算到 hidden_dim 的 h）=====
    h_in = node_emb_fused  # [N, 512] CPU
    N = h_in.size(0)

    # hidden_dim：建议从模型里读出来更稳
    hidden_dim = model.gnn.convs[-1].out_channels

    h_out = torch.empty((N, hidden_dim), dtype=torch.float32, device="cpu")

    node_chunk = 50000  # 内存不够就调小：20000/10000 都行
    for start in tqdm(range(0, N, node_chunk), desc="GNN node chunks"):
        end = min(start + node_chunk, N)

        # 当前要更新的目标节点（全局 id）
        target_nodes_np = np.arange(start, end, dtype=np.int64)

        # 构建 1-hop 子图（features + sub_edge_index）
        sub_x, sub_edge_index, all_nodes, target_mask = get_infer_subgraph(
            target_nodes_np, hop=1, x=h_in
        )

        # 在子图上跑多层 GCNConv
        sub_h = sub_x
        for conv in model.gnn.convs:
            sub_h = conv(sub_h, sub_edge_index)  # 都在 CPU
            sub_h = F.relu(sub_h)

        # 把 target 节点对应的 embedding 写回全局
        # 注意：target_mask 对应 all_nodes 中属于 [start:end] 的节点
        h_out[start:end] = sub_h[target_mask]

        # 释放临时变量（减少峰值）
        del sub_x, sub_edge_index, sub_h, all_nodes, target_mask
        gc.collect()
    # 最终 h 就是全图节点的 hidden embedding
    h = h_out

    # ===== 3.3 对边分块跑 edge_mlp，避免一次性吃爆内存 =====
    edge_index = full_edge_index
    src, dst = edge_index  # [E], [E]
    E_total = src.shape[0]

    chunk_size = 500000  # 可以根据内存调整，比如 200k~1M
    logit_list = []

    print(f"Computing edge probabilities in chunks (chunk_size={chunk_size})...")
    for start in tqdm(range(0, E_total, chunk_size), desc="Edge chunks"):
        end = min(start + chunk_size, E_total)

        # 当前 chunk 的边端点索引
        src_chunk = src[start:end]
        dst_chunk = dst[start:end]

        # 取节点特征
        h_src = h[src_chunk]  # [chunk, hidden_dim]
        h_dst = h[dst_chunk]  # [chunk, hidden_dim]
        h_abs = torch.abs(h_src - h_dst)

        # 拼接成 edge 特征
        h_edge = torch.cat([h_src, h_dst, h_abs], dim=-1)  # [chunk, 3*hidden_dim]

        # MLP 得到 logit
        logit_chunk = model.gnn.edge_mlp(h_edge).squeeze(-1)  # [chunk]

        # 收集到 list，放到 CPU（已经在 CPU 上，但保持一致）
        logit_list.append(logit_chunk.cpu())

    # 拼接所有 chunk 的结果
    logit_all = torch.cat(logit_list, dim=0)  # [E_total]
    prob_all = torch.sigmoid(logit_all)  # [E_total]

# 4. 转换为 Numpy 并保存边概率
alpha = prob_all.numpy()
edge_index_np = edges.T  # [2, E]

# 保存边概率
np.savez(
    #"<EDGE_PROBABILITY_NPZ>",
    "<EDGE_PROBABILITY_NPZ>",
    edge_index=edge_index_np,   # [2, E]
    alpha=alpha,                # [E]
    rows=rows,                  # [N]
    cols=cols                   # [N]
)
print("Edge probabilities saved.")

# 【更改点 2】：保存 MoE 权重
# 创建 MoE 文件夹
os.makedirs("<MOE_WEIGHT_OUTPUT_DIR>", exist_ok=True)
# gates 是 Tensor，转为 Numpy
gates_np = gates.numpy()
# 保存
#np.save("<NUCLEUS_SEGMENTATION_MASK_NPY>", gates_np)
np.save("<MOE_WEIGHTS_NPY>", gates_np)

print(f"MoE weights saved. Shape: {gates_np.shape}")

print("Done!")


df_curve = pd.DataFrame({
    "epoch": epoch_list,
    "loss": loss_list,
    "accuracy": acc_list
})
df_curve.to_csv("<TRAINING_CURVE_CSV>", index=False)
print("<TRAINING_CURVE_CSV>")


plt.figure(figsize=(8, 5))

# Loss（左轴）
plt.plot(epoch_list, loss_list, label="Loss", color="tab:red")
plt.xlabel("Epoch")
plt.ylabel("Loss", color="tab:red")
plt.tick_params(axis="y", labelcolor="tab:red")

# Acc（右轴）
ax2 = plt.gca().twinx()
ax2.plot(epoch_list, acc_list, label="Accuracy", color="tab:blue")
ax2.set_ylabel("Accuracy", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("Training Loss & Accuracy over Epochs")
plt.grid(alpha=0.3)
plt.tight_layout()
out_png = "<TRAINING_CURVE_FIGURE>"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
