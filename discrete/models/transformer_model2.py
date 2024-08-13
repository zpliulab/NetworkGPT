import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from discrete import network_preprocess
from discrete import diffusion_utils
from discrete.models.layers import Xtoy, Etoy, masked_softmax

class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor, E_reg=None, E_LLM=None):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask, E_reg=E_reg, E_LLM=E_LLM)  # Encoder: 1. self-Attention

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)     # Encoder: 2. res_add & normalize layer

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))   # Encoder: 3. Feed Forward (Liner+relu+Liner)
        # ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)     # Encoder: 4. add & normalize layer

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        # ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        # ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y, E_reg, E_LLM


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM E_reg to E
        self.e_reg_add = Linear(de, dx)
        self.e_reg_mul = Linear(de, dx)

        # FiLM E_LLM to E
        self.e_LLM_add = Linear(de, dx)
        self.e_LLM_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask, E_reg=None, E_LLM=None):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df) FiLM(E,Y) = Y .* E1 + Y + E2, where E1 = liner1(E), E2 = liner2(E)

        # 将基于数据库知识的的E_reg的正则化项加入到Y中
        if E_reg is not None:
            zero_mask = E_reg[:, :, :, 0] == 0
            E_reg1 = self.e_reg_mul(E_reg) * e_mask1 * e_mask2  # bs, n, n, dx
            zero_mask1 = zero_mask.unsqueeze(-1).expand(-1, -1, -1, E_reg1.size(-1))
            E_reg1 = torch.where(zero_mask1, torch.zeros_like(E_reg1), E_reg1)
            E_reg1 = E_reg1.reshape((E_reg.size(0), E_reg.size(1), E_reg.size(2), self.n_head, self.df))

            E_reg2 = self.e_reg_add(E_reg) * e_mask1 * e_mask2  # bs, n, n, dx
            zero_mask2 = zero_mask.unsqueeze(-1).expand(-1, -1, -1, E_reg2.size(-1))
            E_reg2 = torch.where(zero_mask2, torch.zeros_like(E_reg2), E_reg2)
            E_reg2 = E_reg2.reshape((E_reg.size(0), E_reg.size(1), E_reg.size(2), self.n_head, self.df))

            Y = Y * (E_reg1 + 1) + E_reg2

        # 将基于大语言模型知识的的E_LLM的正则化项加入到Y中
        if E_LLM is not None:
            E_LLM1 = self.e_reg_mul(E_LLM) * e_mask1 * e_mask2  # bs, n, n, dx
            E_LLM1 = E_LLM1.reshape((E_LLM.size(0), E_LLM.size(1), E_LLM.size(2), self.n_head, self.df))
            E_LLM2 = self.e_reg_add(E_LLM) * e_mask1 * e_mask2  # bs, n, n, dx
            E_LLM2 = E_LLM2.reshape((E_LLM.size(0), E_LLM.size(1), E_LLM.size(2), self.n_head, self.df))

            Y = Y * (E_LLM1 + 1) + E_LLM2

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE         # (bs, n, n, n_head, df) FiLM(y,newE) = newE .* y1 + newE + y2, where E1 = liner1(E), E2 = liner2(E)

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


def tensor_batch_zscore(gene_LLM_embedding):
    # 假设 gene_LLM_embedding 是一个形状为 [20, 187, 512] 的张量
    # 调整张量形状以合并批次和样本维度
    embedding_flat = gene_LLM_embedding.view(-1, gene_LLM_embedding.size(2))

    # 计算均值和标准差
    embedding_mean = embedding_flat.mean(dim=0, keepdim=True)
    embedding_std = embedding_flat.std(dim=0, keepdim=True)

    # 应用Z-Score标准化
    embedding_normalized_flat = (embedding_flat - embedding_mean) / (embedding_std + 1e-5)

    # 如果需要，可以将张量形状恢复到原始的批次维度
    embedding_normalized = embedding_normalized_flat.view(gene_LLM_embedding.shape)

    return embedding_normalized


class DigNetGraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers             # 层的数量
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.hidden_dims = hidden_dims
        self.act_fn_in = act_fn_in

        self.mlp_in_LLM = nn.Sequential(nn.Linear(input_dims['LLM'], hidden_mlp_dims['X']),
                                        act_fn_in,
                                        nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']))

        self.mlp_in_X_Add_LLM = nn.Sequential(nn.Linear(hidden_dims['dx']*2, hidden_mlp_dims['X']),
                                              act_fn_in, nn.Dropout(0.1),
                                              nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']))

        self.mlp_in_X_mul_LLM = nn.Sequential(nn.Linear(hidden_dims['dx']*2, hidden_mlp_dims['X']),
                                              act_fn_in, nn.Dropout(0.1),
                                              nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']))

        self.mlp_in_E_LLM = nn.Sequential(nn.Linear(1, hidden_dims['de']), act_fn_in,
                                          nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_E_reg = nn.Sequential(nn.Linear(1, hidden_dims['de']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, noisy_data):
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['y_t']
        node_mask = noisy_data['node_mask']
        E_reg = noisy_data['E_reg']
        gene_LLM_embedding = noisy_data['gene_LLM_embedding']
        E_LLM = None
        E_reg = None
        if E_reg is not None:
            zero_mask = E_reg == 0
            E_reg = self.mlp_in_E_reg(E_reg)
            extended_mask = zero_mask.expand(-1, -1, -1, E_reg.size(-1))
            E_reg = torch.where(extended_mask, torch.zeros_like(E_reg), E_reg)

        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)                    # 创建一个单位矩阵
        diag_mask = ~diag_mask.type_as(E).bool()    # 单位阵的类型=E，并且对角False，其余True
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)   # 掩码矩阵的维度与E一致

        if gene_LLM_embedding is not None:
            # 方法1，构建相似度网络并融合
            # x_expanded = gene_LLM_embedding.unsqueeze(2)  # 形状变为 [20, 187, 1, 512]
            # x_tiled = gene_LLM_embedding.unsqueeze(1)  # 形状变为 [20, 1, 187, 512]
            # cosine_sim = torch.cosine_similarity(x_expanded, x_tiled, dim=-1)
            # cosine_sim = cosine_sim.unsqueeze(-1)
            # E_LLM = self.mlp_in_E_LLM(cosine_sim)
            X = self.mlp_in_X(X)
            # 方法2，直接相加相乘融合
            # gene_LLM_embedding = tensor_batch_zscore(gene_LLM_embedding)
            # X = tensor_batch_zscore(X)
            # X = 0.95*self.mlp_in_X(X)  # 对基因特征进行处理
            # gene_LLM_embedding = 0.05*self.mlp_in_LLM(gene_LLM_embedding)    # 对LLM特征进行处理
            # X_add = self.mlp_in_X_Add_LLM(torch.cat([X, gene_LLM_embedding], dim=-1))  # 将LLM特征与X特征进行拼接  newX = linear(LLM) + linear(X)
            # X_numl = self.mlp_in_X_mul_LLM(torch.cat([X, gene_LLM_embedding], dim=-1)) # 将LLM特征与X特征进行拼接  newX = linear(LLM) * linear(X)
            # X = self.act_fn_in(X_add + X_numl)  # 将LLM特征与X特征进行拼接  newX = linear(LLM) * linear(X) + linear(LLM) + linear(X)
        else:
            X = self.act_fn_in(self.mlp_in_X(X))

        after_in = network_preprocess.PlaceHolder(X=X, E=self.mlp_in_E(E), y=self.mlp_in_y(y)).mask(node_mask) # 使用掩码矩阵更新X E Y特征（将数据处理成可变长度）

        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y, E_reg, E_LLM = layer(X, E, y, node_mask, E_reg, E_LLM)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        E = E * diag_mask
        pred_probs_E = F.softmax(E, dim=-1)


        return network_preprocess.PlaceHolder(X=X, E=pred_probs_E, y=y).mask(node_mask)
