import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder_overall(Module):

    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dropout = dropout
        self.act = act

        self.encoder_embedding = Cell_embedding_Encoder(512, 64)
        self.decoder_embedding = Cell_embedding_Decoder(64, 512)

        self.encoder_omics1 = DeepEncoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = DeepDecoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)

        self.atten_feature1 = AttentionLayer(self.dim_out_feat_omics1,self.dim_out_feat_omics1)
        self.atten_feature2 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_feature = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)



    def forward(self, features_omics1, adj_spatial_omics1, adj_feature_omics1, cell_embedding, adj_emb):
        # cell_embedding
        embedding_cellembedding_spa = self.encoder_embedding(cell_embedding, adj_spatial_omics1)
        embedding_cellembedding_fea = self.encoder_embedding(cell_embedding, adj_emb)

        # graph1
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)    # torch.Size([3484, 64])

        # graph2
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)

        # feature attention
        embedding_att1, alpha_att1 = self.atten_feature1(embedding_cellembedding_spa, emb_latent_spatial_omics1)
        embedding_att2, alpha_att2 = self.atten_feature2(embedding_cellembedding_fea, emb_latent_feature_omics1)
        embedding_latent_omics1, alpha_att_omics1 = self.atten_feature(embedding_att1, embedding_att2)


        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_omics1(embedding_latent_omics1, adj_spatial_omics1)

        emb_recon_spa = self.decoder_embedding(embedding_cellembedding_spa, adj_spatial_omics1)
        emb_recon_fea = self.decoder_embedding(embedding_cellembedding_fea, adj_emb)


        results = {'emb_latent_omics1': embedding_latent_omics1,
                   'emb_recon_omics1': emb_recon_omics1,
                   'alpha_omics1': alpha_att_omics1,
                   'alpha_att1': alpha_att1,
                   'alpha_att2': alpha_att2,
                   'emb_recon_spa': emb_recon_spa,
                   'emb_recon_fea': emb_recon_fea
                   }

        return results


class Encoder(Module):
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x

class DeepEncoder(Module):
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(DeepEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.hidden_dim = out_feat * 2

        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.hidden_dim))
        self.weight3 = Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.weight2 = Parameter(torch.FloatTensor(self.hidden_dim, self.out_feat))


        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight3)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight1)
        x = torch.spmm(adj, x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.mm(x, self.weight3)
        x = torch.spmm(adj, x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.mm(x, self.weight2)
        x = torch.spmm(adj, x)

        return x

class Cell_embedding_Encoder(Module):
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Cell_embedding_Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x

class Cell_embedding_Decoder(Module):
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Cell_embedding_Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x

class Decoder(Module):
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Reconstructed representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x

class DeepDecoder(Module):
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Reconstructed representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(DeepDecoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.hidden_dim = out_feat * 2

        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.hidden_dim))
        self.weight3 = Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.weight2 = Parameter(torch.FloatTensor(self.hidden_dim, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight3)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight1)
        x = torch.spmm(adj, x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.mm(x, self.weight3)
        x = torch.spmm(adj, x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.mm(x, self.weight2)
        x = torch.spmm(adj, x)

        return x

class AttentionLayer(Module):
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.

    Returns
    -------
    Aggregated representations and modality weights.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

        #torch.nn.init.xavier_normal_(self.w_omega)
        #torch.nn.init.xavier_normal_(self.u_omega)

        #torch.nn.init.kaiming_normal_(self.w_omega, nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(self.u_omega, nonlinearity='relu')

        #torch.nn.init.orthogonal_(self.w_omega)
        #torch.nn.init.orthogonal_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []

        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha

class AttentionLayer_tri(Module):

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer_tri, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)


    def forward(self, emb1, emb2, cell_emb):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(cell_emb), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha

class AttentionLayer_spa_fea(Module):

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer_spa_fea, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2, cell_emb_spa, cell_emb_fea):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(cell_emb_spa), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(cell_emb_fea), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha