import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv   # 也可以用 GraphSAGEConv, GATConv 等

class EdgeGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        # 第一层
        self.convs.append(GCNConv(in_dim, hidden_dim))
        # 后续层
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 边 MLP：输入 [h_i, h_j, |h_i - h_j|] → 标量 logit
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index):
        # 1) GNN 节点更新
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        # 2) 计算每条边的 logit 和 概率
        src, dst = edge_index
        h_src = h[src]      # [E, hidden_dim]
        h_dst = h[dst]      # [E, hidden_dim]
        h_abs = torch.abs(h_src - h_dst)

        h_edge = torch.cat([h_src, h_dst, h_abs], dim=-1)    # [E, 3*hidden_dim]
        logit = self.edge_mlp(h_edge).squeeze(-1)            # [E]
        prob  = torch.sigmoid(logit)                         # [E], α_(i,j) ∈ (0,1)

        return logit, prob


class ModalityMoE(nn.Module):
    def __init__(self, gene_dim=512, img_dim=1024, output_dim=512):
        super(ModalityMoE, self).__init__()
        self.gene_expert = nn.Sequential(nn.Linear(gene_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(),
                                         nn.Linear(output_dim, output_dim))
        self.img_expert = nn.Sequential(nn.Linear(img_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(),
                                        nn.Linear(output_dim, output_dim))
        self.gate = nn.Sequential(nn.Linear(gene_dim + img_dim, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=1))

    def forward(self, x_gene, x_img):
        h_gene = self.gene_expert(x_gene)
        h_img = self.img_expert(x_img)
        weights = self.gate(torch.cat([x_gene, x_img], dim=1))
        return weights[:, 0:1] * h_gene + weights[:, 1:2] * h_img, weights

class EndToEndModel(nn.Module):
    def __init__(self, gene_dim, img_dim, gnn_hidden, gnn_layers):
        super(EndToEndModel, self).__init__()
        self.moe = ModalityMoE(gene_dim, img_dim, output_dim=gene_dim)
        self.gnn = EdgeGNN(in_dim=gene_dim, hidden_dim=gnn_hidden, num_layers=gnn_layers)

    def forward(self, x_gene, x_img, edge_index):
        node_emb_fused, gates = self.moe(x_gene, x_img)
        # GNN 返回 (logit_all, prob_all)
        logit, prob = self.gnn(node_emb_fused, edge_index)
        return logit, prob, gates, node_emb_fused


class EndToEndModelImageOnly(nn.Module):
    def __init__(self, img_dim, gnn_hidden, gnn_layers, proj_dim=512):
        super(EndToEndModelImageOnly, self).__init__()

        # 先把 image embedding 投影到 GNN 输入维度
        self.img_proj = nn.Linear(img_dim, proj_dim)

        # 再送进 GNN
        self.gnn = EdgeGNN(in_dim=proj_dim, hidden_dim=gnn_hidden, num_layers=gnn_layers)

    def forward(self, x_img, edge_index):
        node_emb = self.img_proj(x_img)   # [N, proj_dim]
        logit, prob = self.gnn(node_emb, edge_index)
        return logit, prob, node_emb