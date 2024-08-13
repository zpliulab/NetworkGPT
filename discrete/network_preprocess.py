import torch_geometric.utils
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import torch.nn.functional as F
import numpy as np
from lightgbm import LGBMRegressor
from joblib import Parallel, delayed


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0   # 边特征为0，表示没有这条边，维度为E(0:2)，即batch*edge*edge
    first_elt = E[:, :, :, 1]
    first_elt[no_edge] = 1     # E本身是非0-1的邻接矩阵，这里是把不存在的边的特征修改为 [0,1]  其余的边为[1,0]这种one-hot编码的特征
    E[:, :, :, 1] = first_elt  # [1,0]的为不存在的边，其余的为存在
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def get_max_node(data):
    X, _ = to_dense_batch(x=data.x, batch=data.batch)
    max_num_nodes = X.size(1)
    return max_num_nodes


def to_dense(x, edge_index, edge_attr, batch=None, training=True, max_num_nodes=None, discrete=True):
    """
     -- to_dense_batch是将每个batch中的节点数量转化为相同大小，输出大小为  batch * Max_X * X_attr = 512*9*4
         比如有[1个节点特征,3个节点特征，2个节点特征]三个batch，那么经过转换变成
         [3个节点特征,3个节点特征，3个节点特征]（这里缺失值补0），同时生成node_mask矩阵,[[True,False,False], [True,True,True], [True,True,False]]
     -- to_dense_adj是将每个batch中的稀疏边转化为具有相同相大小的邻接矩阵，输出大小为  batch * Max_Edge * Max_Edge * Edge_attr = 512*9*9*5
    """
    if training:
        X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
       # max_num_nodes = X.size(1)
    else:
        X = x
        node_mask = None
    # node_mask = node_mask.float()
   # edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)  # 移除自环边
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case

    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr.reshape(-1), max_num_nodes=max_num_nodes)
    E_thre = (E > 0).int()
    E_onehot = F.one_hot(E_thre.to(torch.int64), num_classes=2).float()
    if torch.any((E > 0) & (E < 1)):
        mask = E_onehot[..., 1].bool()
        E_onehot[..., 1] = torch.where(mask, E, E_onehot[..., 1])
   # E = encode_no_edge(E)
    if not discrete:
        E_onehot = E.unsqueeze(-1)
    return PlaceHolder(X=X, E=E_onehot, y=None), node_mask, max_num_nodes


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if self.y is not None:
            self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.E = torch.argmax(self.E, dim=-1)  # 沿着这个维度找到具有最大值的索引
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1 #这些边被视为无效或不存在。
        else:
            self.X = self.X * x_mask             # 保留有效节点
            self.E = self.E * e_mask1 * e_mask2  # 保留有效边
           # assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))  # 保证对称性
        return self




from sklearn.tree import BaseDecisionTree
def compute_feature_importances(estimator):
    if isinstance(estimator, BaseDecisionTree):
        return estimator.tree_.compute_feature_importances(normalize=False)
    else:
        importances = [e.tree_.compute_feature_importances(normalize=False)
                       for e in estimator.estimators_]
        importances = np.array(importances)
        return np.sum(importances,axis=0) / len(estimator)



def Random_regression_single(Q, target_feature_idx):
    y = Q[target_feature_idx, :].numpy()
    X = Q[:, :].numpy().T
    if np.sum(y) == 0:
        return np.zeros(Q.shape[0])
    else:

        LGBM = LGBMRegressor(n_estimators=50, max_features='auto', random_state=42, verbose=-1)
        LGBM.fit(X, y)
        # treeEstimator = RandomForestRegressor(n_estimators=50, max_features='auto')
        # treeEstimator.fit(X, y)
        # feature_importances = compute_feature_importances(treeEstimator)
        return LGBM.feature_importances_ / np.sum(LGBM.feature_importances_)

def Random_regression(Q):
    feature_importances_list = []
    Q = Q.cpu()
    for target_feature_idx in range(Q.shape[0]):
        feature_importances = Random_regression_single(Q, target_feature_idx)
        feature_importances = feature_importances / np.sum(feature_importances)
        feature_importances_list.append(feature_importances)
    results = np.stack(feature_importances_list)
    # results = Parallel(n_jobs=-1)(delayed(Random_regression_single)(target_feature_idx)
    #                               for target_feature_idx in range(Q.shape[0]))
    feature_importances_tensor = torch.tensor(np.array(results))
    feature_importances_tensor = torch.nan_to_num(feature_importances_tensor, nan=0.0)
    return feature_importances_tensor


def calculate_similarity(gene_LLM_embedding, LLM_metric='cosine'):
    """
    计算给定张量的相似度或距离。

    参数:
    - embeddings: 形状为 (batch_size, seq_length, embedding_dim) 的张量
    - metric: 使用的度量方法，可以是 'cosine', 'euclidean', 'manhattan', 'pearson', 或 'kobe'

    返回:
    - 相似度或距离矩阵，形状为 (batch_size, seq_length, seq_length)
    """
    x_expanded = gene_LLM_embedding.unsqueeze(1)  # 形状变为 [20, 187, 1, 512]
    x_tiled = gene_LLM_embedding.unsqueeze(0)  # 形状变为 [20, 1, 187, 512]

    if LLM_metric == 'cos':
        # 计算余弦相似度
        similarity = F.cosine_similarity(x_expanded, x_tiled, dim=-1)
        similarity = (similarity + 1) / 2
    elif LLM_metric == 'euclidean':
        # 计算欧氏距离
        difference = x_expanded - x_tiled
        similarity = torch.norm(difference, p=2, dim=-1)
        similarity = 1 / (1 + similarity)
    elif LLM_metric == 'manhattan':
        # 计算曼哈顿距离
        difference = x_expanded - x_tiled
        similarity = torch.sum(torch.abs(difference), dim=-1)  # 负号转换为相似度
        similarity = 1 / (1 + similarity)
    elif LLM_metric == 'tree':
        # 基于Tree计算重要性评分
        similarity = Random_regression(gene_LLM_embedding)
    elif LLM_metric == 'pearson':
        # 归一化输入以零均值
        mean_expanded = x_expanded.mean(dim=-1, keepdim=True)
        mean_tiled = x_tiled.mean(dim=-1, keepdim=True)
        x_expanded_normalized = x_expanded - mean_expanded
        x_tiled_normalized = x_tiled - mean_tiled
        # 计算标准化后的协方差和标准差
        covariance = (x_expanded_normalized * x_tiled_normalized).sum(dim=-1)
        std_expanded = torch.sqrt((x_expanded_normalized ** 2).sum(dim=-1))
        std_tiled = torch.sqrt((x_tiled_normalized ** 2).sum(dim=-1))
        similarity = covariance / (std_expanded * std_tiled)
        similarity = (similarity + 1) / 2
    else:
        raise ValueError("Unsupported metric. Choose 'cosine', 'euclidean', or 'manhattan'.")


    indices = torch.arange(similarity.size(1))  # tensor.size(1) 是每个矩阵的行数或列数
    similarity[indices, indices] = 0
    return similarity
