import numpy as np
import networkx as nx
import itertools
import math
import copy
import pandas as pd
import pingouin as pg
from sklearn.linear_model import LinearRegression


def pca_pcc(data, new_net_bit, theta, max_order, L = -1, show=False):
    predicted_graph = nx.DiGraph()
    predicted_graph.add_nodes_from(data.index.to_list())
    for _, row in new_net_bit.iterrows():
        TF = row['TF']
        Gene = row['Gene']
        predicted_graph.add_edge(TF, Gene)
    num_edges = predicted_graph.number_of_edges()
    nochange = False
    data = data.T
    for u, v,in predicted_graph.edges():
        predicted_graph.edges[u, v]['strength'] = 0
    while L < max_order and nochange == False:
        L = L + 1
        predicted_graph, nochange = remove_edges(predicted_graph, data, L, theta)
    if show:
        print("Final Prediction:")
        print("-----------------")
        print("Order : {}".format(L))
        print("Number of edges in the predicted graph : {}".format(predicted_graph.number_of_edges()))
    predicted_adjMatrix = nx.adjacency_matrix(predicted_graph)
    return predicted_adjMatrix, predicted_graph


def remove_edges(predicted_graph, data, L, theta):
    initial_num_edges = predicted_graph.number_of_edges()
    edges = predicted_graph.edges()

    for edge in list(edges):
        neighbors1 = set(predicted_graph.neighbors(edge[0]))
        neighbors2 = set(predicted_graph.neighbors(edge[1]))
        neighbors = neighbors1.intersection(neighbors2)
        nhbrs = copy.deepcopy(sorted(neighbors))
        T = len(nhbrs)
        if (T < L and L != 0) or edge[0] == edge[1]:
            if edge[0] == edge[1]:
                predicted_graph[edge[0]][edge[1]]['strength'] = 1
            continue
        else:
            x = data[edge[0]].to_numpy()
            if x.ndim == 1:
                x = np.reshape(x, (-1, 1))
            y = data[edge[1]].to_numpy()
            if y.ndim == 1:
                y = np.reshape(y, (-1, 1))
            K = list(itertools.combinations(nhbrs, L))
            if L == 0:
                cmiVal = conditional_pearson_corr(x, y)

                if cmiVal < theta:
                    predicted_graph.remove_edge(edge[0], edge[1])
                else:
                    predicted_graph[edge[0]][edge[1]]['strength'] = cmiVal
            else:
                maxCmiVal = 0
                for zgroup in K:
                    XYZunique = len(np.unique(list([edge[0], edge[1], zgroup[0]])))
                    if XYZunique < 3:
                        continue
                    else:
                        z = data[list(zgroup)].to_numpy()
                        if z.ndim == 1:
                            z = np.reshape(z, (-1, 1))
                        cmiVal = conditional_pearson_corr(x, y, z)
                    if cmiVal > maxCmiVal:
                        maxCmiVal = cmiVal
                if maxCmiVal < theta:
                    predicted_graph.remove_edge(edge[0], edge[1])
                else:
                    predicted_graph[edge[0]][edge[1]]['strength'] = maxCmiVal
    final_num_edges = predicted_graph.number_of_edges()
    if final_num_edges < initial_num_edges:
        return predicted_graph, False
    return predicted_graph, True

def conditional_pearson_corr(X, Y, Z=None):
    if Z is None:
        # 如果没有控制变量，直接计算X和Y的Pearson相关系数
        return np.abs(np.corrcoef(X.T, Y.T)[0, 1])
    else:
        # 控制变量Z存在时，计算X, Y关于Z的条件Pearson相关系数
        # # 使用线性回归的残差
        # lr_ZX = LinearRegression().fit(Z.T, X.T)
        # lr_ZY = LinearRegression().fit(Z.T, Y.T)
        #
        # # 求解残差
        # res_X = X.T - lr_ZX.predict(Z.T)
        # res_Y = Y.T - lr_ZY.predict(Z.T)
        # np.abs(np.corrcoef(res_X, res_Y)[0, 1])
        data = pd.DataFrame({'X': X.flatten(), 'Y': Y.flatten(), 'Z': Z.flatten()})
        partial_corr = pg.partial_corr(data=data, x='X', y='Y', covar='Z')
        # 计算残差的Pearson相关系数
        return np.abs(partial_corr['r'].iloc[0])

