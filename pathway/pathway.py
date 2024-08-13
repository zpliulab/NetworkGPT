import copy
import os
import pickle
import subprocess

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import torch
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from discrete.diffusion_utils import cal_identify_TF_gene
from discrete import network_preprocess

# mapminmax
def MaxMinNormalization(x, Min=0, Max=1):
    x = (x - Min) / (Max - Min)
    return x


# calculate each type percent of edges in GRN
def cal_percent(new_bit_crop, corr_TF_Gene, MI_TF_Gene, net_bit_orig):
    new_bit_crop = new_bit_crop['TF'] + '-' + new_bit_crop['Gene']
    if new_bit_crop.shape[0] == 0:
        return 0, 0, 0
    net_bit_origC = net_bit_orig['TF'] + '-' + net_bit_orig['Gene']
    net_bit_origC = pd.Series(list(set(new_bit_crop) & set(net_bit_origC)))
    NUM_ORIG = net_bit_origC.shape[0] / new_bit_crop.shape[0] * 100

    if len(corr_TF_Gene) > 0:
        corr_TF_GeneC = corr_TF_Gene['TF'] + '-' + corr_TF_Gene['Gene']
        corr_TF_GeneC = pd.Series(list(set(new_bit_crop) & set(corr_TF_GeneC)))
        count_PCC = (~corr_TF_GeneC.isin(net_bit_origC)).sum()
        NUM_PCC = count_PCC / new_bit_crop.shape[0] * 100
    else:
        NUM_PCC = 0

    if len(MI_TF_Gene) > 0:
        MI_TF_GeneC = MI_TF_Gene['TF'] + '-' + MI_TF_Gene['Gene']
        MI_TF_GeneC = pd.Series(list(set(new_bit_crop) & set(MI_TF_GeneC)))
        count_MI = (~MI_TF_GeneC.isin(net_bit_origC)).sum()
        NUM_MI = count_MI / new_bit_crop.shape[0] * 100
    else:
        NUM_MI = 0

    if (NUM_ORIG + NUM_PCC + NUM_MI) != 100:
        SUM1 = (NUM_PCC + NUM_MI)
        NUM_PCC = NUM_PCC * (100 - NUM_ORIG) / SUM1
        NUM_MI = NUM_MI * (100 - NUM_ORIG) / SUM1

    if NUM_PCC + NUM_MI > 50:
        overflow = True
    else:
        overflow = False
    return NUM_ORIG, NUM_PCC, NUM_MI, overflow


def load_sergio_count(filename='pathway/simulation/SERGIO_data_node_2000.data', num=None, logp=True):
    with open(filename, 'rb') as f:  # open file in append binary mode
        batch = pickle.load(f)
    if num is not None:
        x = np.array(batch[num]['exp'])
        if logp:
            x = np.log1p(x)  # 使用 log1p 进行对数变换，避免零值的问题
        batch[num]['exp'] = x
        batch = batch[num]

    return batch


# Plot each type percent of edges in GRN
def plot_GRN_percent(network_percent):
    # 使用布尔索引来删除行和为0的条目
    network_percent = network_percent[network_percent[['NUM_ORIG', 'NUM_PCC', 'NUM_MI']].sum(axis=1) != 0]
    # 设置绘图风格
    sns.set(style="whitegrid")
    # palette = sns.color_palette("coolwarm", 3)
    color = ['#D0AFC4', '#79B99D']
    palette = sns.color_palette("Paired", 11)
    # 绘制堆叠柱状图
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Pathway', y='NUM_ORIG', data=network_percent, color='#EEEEEF',
                label='NUM_ORIG', edgecolor='none', dodge=False)
    sns.barplot(x='Pathway', y='NUM_PCC', data=network_percent, color='#173565',
                label='NUM_PCC', bottom=network_percent['NUM_ORIG'], edgecolor='none', dodge=False)
    sns.barplot(x='Pathway', y='NUM_MI', data=network_percent, color='#D99943',
                label='NUM_MI', bottom=network_percent['NUM_ORIG'] + network_percent['NUM_PCC'], edgecolor='none',
                dodge=False)
    # 添加图例
    plt.legend()

    # 添加标签和标题
    plt.xlabel('Pathway')
    plt.ylabel('percent')
    plt.title('GRN percent')
    #  plt.show()
    # 显示图形
    plt.savefig("Train_GRN_pecent_bar.pdf")


# cal MI
def compute_mutual_information(df):
    features = df.values
    num_rows = len(features)
    mi_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            mi = mutual_info_score(features[i], features[j])
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    mi_matrix = pd.DataFrame(mi_matrix, index=df.index, columns=df.index)
    return mi_matrix


def compare_char(charlist, setlist):
    try:
        index = setlist.index(charlist)
    except ValueError:
        index = None
    return index


def calRegnetwork(human_network, GRN_GENE_symbol):
    human_network = human_network[
        human_network['TF'].isin(GRN_GENE_symbol) & human_network['Gene'].isin(GRN_GENE_symbol)]
    human_network_TF_symbol = np.array(human_network['TF'])
    human_network_Gene_symbol = np.array(human_network['Gene'])

    if 'Score' in human_network.columns:
        human_network['Key'] = human_network['TF'] + '-' + human_network['Gene']
        Score_dict = pd.Series(human_network['Score'].values, index=human_network['Key']).to_dict()

    d = 1
    network = []

    for i in range(len(GRN_GENE_symbol)):
        number = [j for j, x in enumerate(human_network_TF_symbol) if str(GRN_GENE_symbol[i]) == x]
        if len(number) > 0:
            for z in range(len(number)):
                networkn = []
                number2 = compare_char(str(human_network_Gene_symbol[number[z]]), GRN_GENE_symbol)
                if number2 is not None:
                    networkn.append(GRN_GENE_symbol[i])  # 调控基因
                    networkn.append(GRN_GENE_symbol[number2])  # 靶基因
                    if 'Score' in human_network.columns:
                        networkn.append(
                            Score_dict[GRN_GENE_symbol[i] + '-' + GRN_GENE_symbol[number2]])  # Score for TF-Gene
                    network.append(networkn)
                    d += 1
    if 'Score' in human_network.columns:
        network = pd.DataFrame(network, columns=['TF', 'Gene', 'Score'])
    else:
        network = pd.DataFrame(network, columns=['TF', 'Gene'])

    return network


def load_KEGG(kegg_file='pathway/kegg/KEGG_all_pathway.pkl'):
    '''
        load kegg pathway
    '''
    if os.path.exists(kegg_file):
        # 如果 pkl 文件存在，则加载它
        with open(kegg_file, 'rb') as f:
            KEGG = pickle.load(f)
    else:
        # 如果 pkl 文件不存在，则运行 KEGG.py 文件
        subprocess.call(['python', 'pathway/kegg/KEGG_process.py'])
        # 加载生成的 pkl 文件
        with open(kegg_file, 'rb') as f:
            KEGG = pickle.load(f)
    return KEGG


# add high MI
def high_MI(exp_pca_discretized, exp_pca, net_bit, parm):
    row_MI = compute_mutual_information(exp_pca_discretized)
    np.fill_diagonal(row_MI.to_numpy(), 0)
    # MI_thrd = 0.3
    expi = 1
    MI_thrd = (np.exp((expi * 0.01) ** 2) - 1)
    rflag = 1
    while rflag == 1:
        indices = np.where(row_MI > MI_thrd)
        # if parm['MI_percent'] * len(indices[0]) > net_bit.shape[0]:
        #     MI_thrd = MI_thrd + 0.1
        if len(indices[0]) > exp_pca.shape[0] / parm['MI_percent']:
            expi += 1
            MI_thrd = (np.exp((expi * 0.01) ** 2) - 1)
            rflag = 1
        else:
            MI_TF = exp_pca.index[indices[0]]
            MI_Gene = exp_pca.index[indices[1]]
            MI_TF_Gene = pd.DataFrame([MI_TF, MI_Gene], index=['TF', 'Gene']).T
            net_bit = pd.concat([net_bit, MI_TF_Gene], axis=0).reset_index(drop=True)
            net_bit = net_bit.drop_duplicates()
            rflag = 0
    return net_bit, MI_TF_Gene


# add high Pearson corrlation
def high_pearson(exp_pca, net_bit, parm):
    row_corr = exp_pca.T.corr(method='pearson')
    np.fill_diagonal(row_corr.to_numpy(), 0)
    # pearson_thrd = 0.95
    expi = 1
    pearson_thrd = (np.exp((expi * 0.01) ** 2) - 1)
    rflag = 1
    while rflag == 1:
        indices = np.where(row_corr > pearson_thrd)
        #   if parm['pear_percent'] * len(indices[0]) > net_bit.shape[0]:
        #    pearson_thrd = pearson_thrd + 0.0005
        if len(indices[0]) > exp_pca.shape[0] / parm['pear_percent']:
            expi += 1
            pearson_thrd = (np.exp((expi * 0.01) ** 2) - 1)
            rflag = 1
        else:
            corr_TF = exp_pca.index[indices[0]]
            corr_Gene = exp_pca.index[indices[1]]
            corr_TF_Gene = pd.DataFrame([corr_TF, corr_Gene], index=['TF', 'Gene']).T
            net_bit = pd.concat([net_bit, corr_TF_Gene], axis=0).reset_index(drop=True)
            net_bit = net_bit.drop_duplicates()
            rflag = 0
    return net_bit, corr_TF_Gene


def from_cancer_create(BRCA_exp_filter_saver, KEGG, parm, lim=200, test_pathway=None, Other_Pathway=None,
                       human_network=None, TF_file='GRN/TF.txt'):
    if test_pathway is not None:
        BRCA_exp_filter_NoBRCA = BRCA_exp_filter_saver.loc[~BRCA_exp_filter_saver.index.isin(KEGG[test_pathway])]
        exp = BRCA_exp_filter_NoBRCA.loc[BRCA_exp_filter_NoBRCA.index.isin(KEGG[Other_Pathway])]
    else:
        if Other_Pathway[:3] == "hsa":  # test belong to KEGG database
            exp = BRCA_exp_filter_saver.loc[BRCA_exp_filter_saver.index.isin(KEGG[Other_Pathway])]
        else:
            user_define = pd.read_csv(Other_Pathway)
            exp = BRCA_exp_filter_saver.loc[BRCA_exp_filter_saver.index.isin(user_define.iloc[:, -1].tolist())]
    if lim is None:
        lim = 200
    if exp.shape[0] < 10 or exp.shape[0] > lim:
        return None, None, None

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    exp = pd.DataFrame(scaler.fit_transform(exp), columns=exp.columns, index=exp.index)
    net_bit = calRegnetwork(human_network, exp.index.to_list())
    net_bit_orig = copy.deepcopy(net_bit)

    # pro-process data
    # pca = PCA()
    # exp_pca = pd.DataFrame(pca.fit_transform(exp), index=exp.index)
    # exp_pca = exp_pca.drop(exp_pca.columns[-1], axis=1)
    exp_pca = exp
    exp_pca_discretized = pd.DataFrame()
    num_bins = 256
    for column in exp_pca.columns:
        bins = np.linspace(exp_pca[column].min(), exp_pca[column].max(), num_bins + 1)
        #      bins = exp_pca[column].quantile(q=np.linspace(0, 1, num_bins + 1))  # 根据分位数生成等频的区间
        labels = range(num_bins)
        if np.sum(exp_pca[column]) == 0:
            exp_pca_discretized[column] = exp_pca[column]
        else:
            exp_pca_discretized[column] = pd.cut(exp_pca[column], bins=bins, labels=labels, include_lowest=True)  # 执行离散化

    # add high link
    net_bit, corr_TF_Gene = high_pearson(exp_pca, net_bit, parm)
    net_bit, MI_TF_Gene = high_MI(exp_pca_discretized, exp_pca, net_bit, parm)
    print(
        f'***********************      Reg: {net_bit_orig.shape[0]}           pearson: {corr_TF_Gene.shape[0]}              MI: {MI_TF_Gene.shape[0]}')

    if net_bit.shape[1] < 1:
        return None, None, None

    # creat adj
    nodes = np.unique(exp.index)
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for _, row in net_bit.iterrows():
        i = np.where(nodes == row['TF'])[0][0]
        j = np.where(nodes == row['Gene'])[0][0]
        adj_matrix[i, j] = 1

    GENE_ID_list, TF_ID_list = cal_identify_TF_gene(exp.index, TF_file=TF_file)
    for TF_ID in TF_ID_list:
        for GENE_ID in GENE_ID_list:
            adj_matrix[GENE_ID, TF_ID] = 0  # Gene -> TF is error

    predicted_adj_matrix, new_graph = pca_cmi(exp_pca_discretized, net_bit, parm['pmi_percent'], 1)
    predicted_adj_matrix = predicted_adj_matrix.toarray()
    new_bit_crop = pd.DataFrame(new_graph.edges(), columns=['TF', 'Gene'])
    if np.sum(predicted_adj_matrix) == 0:
        new_row = {'Pathway': Other_Pathway, 'NUM_ORIG': 0, 'NUM_PCC': 0, 'NUM_MI': 0}
        return None, None, new_row
    elif (np.sum(adj_matrix) / np.sum(predicted_adj_matrix)) < 0.5:
        NUM_ORIG, NUM_PCC, NUM_MI, overflow = cal_percent(net_bit_orig, corr_TF_Gene, MI_TF_Gene, net_bit_orig)
        new_row = {'Pathway': Other_Pathway, 'NUM_ORIG': NUM_ORIG, 'NUM_PCC': NUM_PCC, 'NUM_MI': NUM_MI}
        return exp, adj_matrix, new_row
    else:
        NUM_ORIG, NUM_PCC, NUM_MI, overflow = cal_percent(new_bit_crop, corr_TF_Gene, MI_TF_Gene, net_bit_orig)
        new_row = {'Pathway': Other_Pathway, 'NUM_ORIG': NUM_ORIG, 'NUM_PCC': NUM_PCC, 'NUM_MI': NUM_MI}
        return exp, predicted_adj_matrix, new_row


def find_entrez_id(genome, genename):
    matching_row = genome[genome['symbol'] == genename]
    if not matching_row.empty:
        # 如果找到匹配的行，则输出对应的 'entrez_id'
        entrez_id = matching_row['entrez_id'].iloc[0]
        return entrez_id
    else:
        return -1


# 加载数据库知识
def return_database(database='RegNetwork', species='human'):
    if database == 'RegNetwork':
        # 加载RegNetwork数据库
        if species == 'human':
            Regnetwork_path = 'pathway/Regnetwork/2022.human.source'
        elif species == 'mouse':
            Regnetwork_path = 'pathway/Regnetwork/2022.mouse.source'
        else:
            print('Species error!')
        dtypes = {1: str, 3: str}
        human_network = pd.read_csv(Regnetwork_path, sep='\t', header=None, dtype=dtypes)
        human_network.columns = ['TF', 'TF_ID', 'Gene', 'Gene_ID']
    elif database == 'KEGG':
        #  加载KEGG数据库
        if species == 'human':
            KEGG_path = 'pathway/kegg/KEGG_human_network_2022.csv'
        elif species == 'mouse':
            KEGG_path = 'pathway/kegg/KEGG_mouse_network_2022.csv'
        else:
            print('Species error!')
        human_network = pd.read_csv(KEGG_path, sep=',', index_col=0)
        human_network.columns = ['TF', 'Gene', 'Repression', 'id']

    elif database == 'STRING':
        #  加载KEGG数据库
        if species == 'human':
            STRING_ID_path = 'pathway/STRING/9606.protein.aliases.v12.0.txt'
            STRING_Link_path = 'pathway/STRING/9606.protein.links.v12.0.txt'
            Genome_path = 'GRN/Genome.txt'
            dtypes = {0: str, 1: str}
            Genome_ID = pd.read_csv(Genome_path, sep='\t', dtype=str)

        elif species == 'mouse':
            STRING_ID_path = 'pathway/STRING/10090.protein.aliases.v12.0.txt'
            STRING_Link_path = 'pathway/STRING/10090.protein.links.v12.0.txt'
            Genome_path = 'GRN/MRK_List2.rpt'
            Genome_ID = pd.read_csv(Genome_path, sep='\t', dtype=str)
            Genome_ID = Genome_ID[Genome_ID['Marker Type'] == 'Gene']
            Genome_ID['symbol'] = Genome_ID['Marker Symbol']
        else:
            print('Species error!')

        protein_ID = pd.read_csv(STRING_ID_path, sep='\t', dtype=dtypes)
        protein_ID = protein_ID.loc[protein_ID['alias'].isin(Genome_ID['symbol'])]  # 只保留有重要意义的基因组
        protein_ID = protein_ID.drop_duplicates(subset=['#string_protein_id'])  # 去除重复值

        human_network = pd.read_csv(STRING_Link_path, sep=' ')
        human_network = human_network.loc[human_network['protein1'].isin(protein_ID['#string_protein_id'])]
        human_network = human_network.loc[human_network['protein2'].isin(protein_ID['#string_protein_id'])]

        # protein ENSP转symbol
        ENSP_to_gene = protein_ID.set_index('#string_protein_id')['alias'].to_dict()  # 制作一个查询字典
        human_network['protein1'] = human_network['protein1'].map(ENSP_to_gene)
        human_network['protein2'] = human_network['protein2'].map(ENSP_to_gene)
        human_network.columns = ['TF', 'Gene', 'Score']
        human_network = human_network.loc[human_network['Score'] > 800]

    elif database == 'TRRUST':
        #  加载TRRUST数据库
        if species == 'human':
            TRRUST_path = 'pathway/TRRUST/trrust_rawdata.human.tsv'
        elif species == 'mouse':
            TRRUST_path = 'pathway/TRRUST/trrust_rawdata.mouse.tsv'
        else:
            print('Species error!')
        human_network = pd.read_csv(TRRUST_path, sep='\t', header=None)
        human_network.columns = ['TF', 'Gene', 'Repression', 'id']

    else:
        raise ValueError('database is not supported')

    return human_network


from functools import lru_cache


@lru_cache(maxsize=None)
def cached_return_database(database):
    return return_database(database)


def load_database(node_feature, theta):
    database = theta['database']
    if database == 'simulation':
        Regnetwork_adj_matrix = theta['simu_pertubed']
        indices = np.where(theta['simu_pertubed'])
        simu_TF = node_feature[indices[0]]
        simu_Gene = node_feature[indices[1]]
        net_bit = pd.DataFrame([simu_TF, simu_Gene], index=['TF', 'Gene']).T

    else:
        # 加载人类基因名
        hunman_genome_list = pd.read_csv('GRN/Genome.txt', sep='\t', dtype=str)
        hunman_genome_list = hunman_genome_list[['symbol', 'ensembl_gene_id', 'entrez_id']]
        entrez_id = []
        assert node_feature.dtype != int
        for genename in node_feature:
            entrez_id.append(find_entrez_id(hunman_genome_list, genename))
        genename = node_feature.to_list()

        # database 分多种情况讨论，如果databse是一个list变量
        if isinstance(database, list):
            human_network_list = []
            for str_base in database:
                human_network_list.append(cached_return_database(database=str_base))
            filtered_dfs = [df[['TF', 'Gene']] for df in human_network_list if 'TF' in df and 'Gene' in df]
            human_network = pd.concat(filtered_dfs, axis=0)
        elif isinstance(database, str):
            human_network = cached_return_database(database=database)
        else:
            raise ValueError('database is not supported')

        net_bit = calRegnetwork(human_network, genename)
        genename = np.array(genename)
        Regnetwork_adj_matrix = np.zeros((len(genename), len(genename)), dtype=int)
        for _, row in net_bit.iterrows():
            i = np.where(genename == row['TF'])[0][0]
            j = np.where(genename == row['Gene'])[0][0]
            Regnetwork_adj_matrix[i, j] = 1
        GENE_ID_list, TF_ID_list = cal_identify_TF_gene(node_feature)
        for TF_ID in TF_ID_list:
            for GENE_ID in GENE_ID_list:
                Regnetwork_adj_matrix[GENE_ID, TF_ID] = 0  # Gene -> TF is error
    return Regnetwork_adj_matrix, net_bit


def delte_low_corr(net_bit, node_feature):
    thr = 0.05
    correlation = node_feature.T.corr(method='pearson')
    # 如果链接关系表的数量超过节点数量的10倍
    while net_bit.shape[0] > 10 * node_feature.shape[0]:
        net_bit = net_bit[net_bit.apply(lambda edge: correlation.loc[edge[0], edge[1]] >= thr, axis=1)]
        thr += 0.05
    return net_bit


def fill_nan2PCC(adj_matrix, node_feature):
    correlation_matrix = node_feature.T.corr(method='pearson')
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if np.isnan(adj_matrix[i, j]):
                adj_matrix[i, j] = correlation_matrix.iloc[i, j]
            if np.isnan(adj_matrix[i, j]):
                adj_matrix[i, j] = 0.0001
    return adj_matrix


# 计算PC-PMI, PC-PCC
def cal_pmi_ppc(node_feature, Regnetwork_adj_matrix, net_bit, per_theta, max_order=1, all_edge_thr=1, show=False):
    """
    node_feature: 基因表达数据
    Regnetwork_adj_matrix: 基因调控网络的邻接矩阵
    net_bit: 基因调控网络的边
    per_theta: PMI和PPC的阈值，当数值<1时，则表示为阈值，否则取百分比的边
    max_order: 最大阶数
    show: 是否显示输出信息
    """
    theta = {}
    theta['PMI'] = 1
    theta['PPC'] = 1
    PMIflag_while = True
    PPCflag_while = True
    Sum_adj = 1e4
    net_bit = delte_low_corr(net_bit, node_feature)  # 如果边太多，则预先删除一下
    Sum_adj_pmi = Sum_adj_ppc = 1e4
    expi = 2

    while expi <= 15 and Sum_adj > all_edge_thr * node_feature.shape[0]:
        theta['PPC'] = (np.exp((expi * 0.01) ** 2) - 1) * 10 if per_theta['PPC'] > 1 else per_theta['PPC']
        theta['PMI'] = (np.exp((expi * 0.01) ** 2) - 1) * 100 if per_theta['PMI'] > 1 else per_theta['PMI']

        if Sum_adj_pmi > all_edge_thr * node_feature.shape[0] or PMIflag_while:
            binary_adjMatrix_pmi, predicted_graph_pmi = pca_cmi(node_feature, net_bit, theta=theta['PMI'],
                                                                max_order=max_order, L=-1, show=show)
            binary_adjMatrix_pmi = binary_adjMatrix_pmi.toarray()
            binary_adjMatrix_pmi[binary_adjMatrix_pmi != 0] = 1
            Sum_adj_pmi = np.sum(np.multiply(Regnetwork_adj_matrix, binary_adjMatrix_pmi))
            PMIflag_while = not PMIflag_while

        if Sum_adj_ppc > all_edge_thr * node_feature.shape[0] or PPCflag_while:
            binary_adjMatrix_ppc, predicted_graph_ppc = pca_pcc(node_feature, net_bit, theta=theta['PPC'],
                                                                max_order=max_order, show=show)
            binary_adjMatrix_ppc = binary_adjMatrix_ppc.toarray()
            binary_adjMatrix_ppc[binary_adjMatrix_ppc != 0] = 1
            Sum_adj_ppc = np.sum(np.multiply(Regnetwork_adj_matrix, binary_adjMatrix_ppc))
            PPCflag_while = not PPCflag_while

        binary_adjMatrix_ppc_pmi = np.multiply(np.multiply(Regnetwork_adj_matrix, binary_adjMatrix_ppc),
                                               binary_adjMatrix_pmi)
        Sum_adj = np.sum(binary_adjMatrix_ppc_pmi)
        overlap_pcc_pmi = round(Sum_adj / np.sum(Regnetwork_adj_matrix), 2) * 100

        expi = expi + 1

        if Sum_adj_pmi <= all_edge_thr * node_feature.shape[0] and Sum_adj_ppc <= all_edge_thr * node_feature.shape[0]:
            break
        if Sum_adj_pmi <= all_edge_thr * node_feature.shape[0] and not PPCflag_while:
            break
        if Sum_adj_ppc <= all_edge_thr * node_feature.shape[0] * 1.5 and not PMIflag_while:
            break
        if overlap_pcc_pmi <= per_theta['PMI'] or overlap_pcc_pmi <= per_theta['PPC']:
            break

    # 计算邻接矩阵
    adj_matrix_with_pmi = nx.to_numpy_array(predicted_graph_pmi, weight='strength')
    adj_matrix_with_pmi = fill_nan2PCC(adj_matrix_with_pmi, node_feature) if np.isnan(
        adj_matrix_with_pmi).any() else adj_matrix_with_pmi  # 填补NAN值
    if np.max(adj_matrix_with_pmi) != 0:
        adj_matrix_with_pmi = adj_matrix_with_pmi / np.max(adj_matrix_with_pmi)  # 归一化
    adj_matrix_with_ppc = nx.to_numpy_array(predicted_graph_ppc, weight='strength')
    adj_matrix_with_ppc = fill_nan2PCC(adj_matrix_with_ppc, node_feature) if np.isnan(
        adj_matrix_with_ppc).any() else adj_matrix_with_ppc
    if np.max(adj_matrix_with_ppc) != 0:
        adj_matrix_with_ppc = adj_matrix_with_ppc / np.max(adj_matrix_with_ppc)

    adj_matrix_with_pmi = np.multiply(binary_adjMatrix_pmi, adj_matrix_with_pmi)
    adj_matrix_with_ppc = np.multiply(binary_adjMatrix_ppc, adj_matrix_with_ppc)

    assert np.all((binary_adjMatrix_pmi == 1) <= (adj_matrix_with_pmi != 0))
    assert np.all((binary_adjMatrix_ppc == 1) <= (adj_matrix_with_ppc != 0))
    adj_pmi_ppc_mean = (adj_matrix_with_pmi + adj_matrix_with_ppc) / 2
    assert not np.isnan(adj_pmi_ppc_mean).any()
    return binary_adjMatrix_pmi, binary_adjMatrix_ppc, adj_pmi_ppc_mean


# 从LLM大模型嵌入文件中加载数据
def load_LLM_embedding(LLM_filepath, node_feature):
    if LLM_filepath[-4:] == 'data':
        with open(LLM_filepath, 'rb') as f:
            LLM_embedding = pickle.load(f)
        hunman_genome_list = pd.read_csv('/home/wcy/Diff_comple/GRN/Genome.txt',
                                         sep='\t', dtype=str)  # 从 Download_TF_file.py 下载的文件
        hunman_genome_list = pd.Series(hunman_genome_list['symbol'])
        index = hunman_genome_list.isin(node_feature)
        index2 = hunman_genome_list[index].index.tolist()
        LLM_embedding = LLM_embedding[index2, :]
    elif LLM_filepath[-3:] == 'csv':
        LLM_embedding = pd.read_csv(LLM_filepath, index_col=0)
        # LLM_embedding = LLM_embedding.loc[node_feature]
        result = pd.DataFrame(index=node_feature, columns=LLM_embedding.columns).fillna(0)
        for idx in node_feature:
            if idx in LLM_embedding.index:
                result.loc[idx] = LLM_embedding.loc[idx]
            else:
                result.loc[idx] = np.zeros(len(LLM_embedding.columns))
                print(f'基因 {idx} : 并不在scFoundation预训练嵌入中，使用全0填充！')
        LLM_embedding = torch.tensor(result.values, dtype=torch.float)
    else:
        print('The file format is not supported! Please check the file format!')

    return LLM_embedding


# 辅助函数： 把邻接矩阵转换为张量
def adj2tensor(adj_pmi_ppc_mean, sc_pc_adj_matrix=None):
    if sc_pc_adj_matrix is None:
        sc_pc_adj_matrix = adj_pmi_ppc_mean
    sc_pc_adj_matrix = torch.tensor(sc_pc_adj_matrix, dtype=torch.float32)
    adj_pmi_ppc_mean = torch.tensor(adj_pmi_ppc_mean, dtype=torch.float32)
    crop_indices_tensor = sc_pc_adj_matrix.nonzero(as_tuple=False).t().contiguous()  # 从张量sc_pc_adj_matrix中提取非零值的索引
    crop_indices_values = adj_pmi_ppc_mean[
        crop_indices_tensor[0, :], crop_indices_tensor[1, :]]  # 从张量adj_pmi_ppc_mean中提取非零值
    if len(crop_indices_values) == 0:
        crop_indices_tensor = torch.tensor([[0], [0]])
        crop_indices_values = torch.tensor([0])
    return crop_indices_tensor, crop_indices_values


def matrix2Data(adj_matrix,
                node_feature,
                num=0,
                adj2data=True,
                log_trans=False,
                metacell=False,
                Cnum=100,
                k=20,
                theta=None,
                Adddatabse=False,
                LLM_filepath=None):
    if theta is None:
        theta = {'PMI': 30, 'PPC': 0.3, 'database': 'RegNetwork'}
    if isinstance(node_feature, list):
        if metacell:
            print(f'Gene: {str(node_feature[num].shape[0])}, Cell: {str(node_feature[num].shape[1])}'
                  f',Start calculating Meta-cell! num = {str(Cnum)}, k = {str(k)}')
            node_feature = cal_metacell(node_feature[num], Cnum=Cnum, k=k)
        x = torch.tensor(np.array(node_feature), dtype=torch.float)
    else:
        if metacell:
            print(f'Gene: {str(node_feature.shape[0])}, Cell: {str(node_feature.shape[1])}'
                  f',Start calculating Meta-cell! num = {str(Cnum)}, k = {str(k)}')
            node_feature = cal_metacell(node_feature, Cnum=Cnum, k=k)
        x = torch.tensor(np.array(node_feature), dtype=torch.float)

    # 对数变换
    if log_trans:
        x = torch.log1p(x)  # 使用 log1p 进行对数变换，避免零值的问题

    # 归一化
    scaler = MinMaxScaler()
    x_normalized = torch.tensor(scaler.fit_transform(x), dtype=torch.float)
    if adj2data:
        if adj_matrix is not None:
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            indices_tensor = adj_matrix.nonzero(as_tuple=False).t().contiguous()
            num_edges = indices_tensor.shape[1]
            values_tensor = torch.ones(num_edges, dtype=torch.float32)
        else:
            indices_tensor = None
            values_tensor = None
    else:
        indices = []
        values = []
        for index, value in np.ndenumerate(adj_matrix):
            indices.append(index)
            values.append(value)
        # 数据类型转化
        indices = np.array(indices).T
        values = np.array(values)
        indices_tensor = torch.tensor(indices, dtype=torch.int64)
        values_tensor = torch.tensor(values, dtype=torch.float)

    # 增加一些其他网络/相关性信息，以便于后续使用
    # 构建一个空tensor
    LLM_embedding = None
    Regnetwork_indices_tensor = None
    Regnetwork_indices_values = None
    crop_indices_tensor = None
    crop_indices_values = None
    LLM_indices_tensor = None
    LLM_indices_values = None
    entrez_id = None
    if Adddatabse:
        Regnetwork_adj_matrix, net_bit = load_database(node_feature.index, theta=theta)
        all_edge_thr = 1.5 if theta['database'] == 'simulation' else 1
        binary_adj_pmi, binary_adj_ppc, adj_pmi_ppc_mean = cal_pmi_ppc(pd.DataFrame(x_normalized,
                                                                                    index=node_feature.index,
                                                                                    columns=node_feature.columns),
                                                                       Regnetwork_adj_matrix,
                                                                       net_bit,
                                                                       theta,
                                                                       all_edge_thr=all_edge_thr,
                                                                       show=False)
        sc_pc_adj_matrix = np.multiply(np.multiply(Regnetwork_adj_matrix, binary_adj_pmi), binary_adj_ppc)
        overlap_pc = round(np.sum(sc_pc_adj_matrix) / np.sum(Regnetwork_adj_matrix), 2) * 100
        print(
            f' 从知识库中找到TF-Gene链接以引导生成 by sc-PC：{overlap_pc} % '
            f'(关联数量--{np.sum(sc_pc_adj_matrix)})(基因数量--{node_feature.shape[0]})')
        assert np.all((sc_pc_adj_matrix == 1) <= (adj_pmi_ppc_mean != 0))
        Regnetwork_indices_tensor, Regnetwork_indices_values = adj2tensor(Regnetwork_adj_matrix)
        crop_indices_tensor, crop_indices_values = adj2tensor(adj_pmi_ppc_mean, sc_pc_adj_matrix=sc_pc_adj_matrix)
        assert crop_indices_tensor.shape[1] == crop_indices_values.shape[0], "sc-PC+先验网络：边的数量与权重出现错误！"
        assert crop_indices_tensor.max() < x_normalized.shape[0], "sc-PC+先验网络：边的最大索引超出表达谱范围！"
    # 添加大模型嵌入
    # print(f'大语言模型中加载基因嵌入参数！')
    if LLM_filepath is not None:
        LLM_embedding = load_LLM_embedding(LLM_filepath, node_feature.index)
        similarity = network_preprocess.calculate_similarity(LLM_embedding, LLM_metric=theta['LLM_metric'])
        LLM_indices_tensor, LLM_indices_values = adj2tensor(similarity)
        assert LLM_embedding.shape[0] == node_feature.shape[0] == x_normalized.shape[0], "LLM embedding维度大小出现错误！"
        assert LLM_indices_tensor.shape[1] == LLM_indices_values.shape[0], "LLM网络：边的数量与权重出现错误！"
        assert LLM_indices_tensor.max() < x_normalized.shape[0], "LLM网络：边的最大索引超出表达谱范围！"

    # # 记录哪些特征是原始特征，哪些特征是LLM_embedding
    # feature_origin_mask_x = torch.ones(x_normalized.shape[1], dtype=torch.bool)
    # # 如果LLM_embedding为空，则不添加
    # if LLM_embedding is not None:
    #     x_cat = x_normalized
    #     feature_origin_mask_L = torch.tensor([], dtype=torch.bool)
    # else:
    #     x_cat = torch.cat((x_normalized, LLM_embedding), dim=1)  # 拼接特征
    #     feature_origin_mask_L = torch.zeros(LLM_embedding.shape[1], dtype=torch.bool)
    # # 记录哪些特征是原始特征，哪些特征是LLM_embedding
    # feature_origin_mask = torch.cat((feature_origin_mask_x, feature_origin_mask_L))
    assert indices_tensor.shape[1] == values_tensor.shape[0], "Gold standard：边的数量与权重出现错误！"
    assert Regnetwork_indices_tensor.shape[1] == Regnetwork_indices_values.shape[0], "先验网络：边的数量与权重出现错误！"
    assert indices_tensor.max() < x_normalized.shape[0], "Gold standard：边的最大索引超出表达谱范围！"
    assert Regnetwork_indices_tensor.max() < x_normalized.shape[0], "先验网络：边的最大索引超出表达谱范围！"

    data = Data(x=x_normalized,
                # feature_mask=feature_origin_mask,
                LLM_embedding=LLM_embedding,
                edge_index=indices_tensor,
                edge_attr=values_tensor,
                Reg_edge_index=Regnetwork_indices_tensor,  # 存储知识库中的边索引
                Reg_edge_attr=Regnetwork_indices_values,  # 存储知识库中的边权重
                crop_Reg_edge_index=crop_indices_tensor,  # 存储修剪的知识库中的边索引
                crop_Reg_edge_attr=crop_indices_values,  # 存储修剪的知识库中的边权重
                llm_edge_index=LLM_indices_tensor,       # 存储LLM中的边索引
                llm_edge_attr=LLM_indices_values,        # 存储LLM中的边权重
                # entrezid=entrez_id,
                y=pd.DataFrame(node_feature).index)

    return data


# Create train/test sets from SEIGIO simulation datasets
def create_batch_dataset_simu(filename='./pathway/simulation/SERGIO_data_node_2000.data', num=None, device=None,
                              test=False, adddata=None, metacell=True, Cnum=100, k=20, Adddatabse=True, theta=None):
    if test:
        with open(filename, 'rb') as f:  # open file in append binary mode
            batch = pickle.load(f)
        if num is not None:
            assert 'exp' in batch[num], "The input pickle file must contain the 'exp' item!"
            assert 'net' in batch[num], "The input pickle file must contain the 'exp' item!"
            theta['database'] = 'simulation' if 'perturbed_net' in batch[num] else False
            theta['simu_pertubed'] = batch[num]['net'] if 'perturbed_net' in batch[num] else False
            Adddatabse = Adddatabse if 'perturbed_net' in batch[num] else False
            batch = matrix2Data(batch[num]['net'],
                                batch[num]['exp'],
                                metacell=metacell,
                                Cnum=Cnum,
                                k=k,
                                log_trans=True,
                                theta=theta,
                                Adddatabse=Adddatabse).to(device)
        return batch
    else:
        data_list = []
        with open(filename, 'rb') as f:  # open file in append binary mode
            data = pickle.load(f)
        edge_percent = []
        for idx, net_exp in enumerate(data):
            if num is not None:
                if num <= idx:
                    break
            assert 'exp' in net_exp, "The input pickle file must contain the 'exp' item!"
            assert 'net' in net_exp, "The input pickle file must contain the 'exp' item!"

            theta['database'] = 'simulation' if 'perturbed_net' in net_exp else False
            theta['simu_pertubed'] = net_exp['perturbed_net'] if 'perturbed_net' in net_exp else False
            Adddatabse = Adddatabse if 'perturbed_net' in net_exp else False
            data_net_exp = matrix2Data(net_exp['net'],
                                       net_exp['exp'],
                                       metacell=metacell,
                                       Cnum=Cnum,
                                       k=k,
                                       log_trans=True,
                                       theta=theta,
                                       Adddatabse=Adddatabse).to(device)
            data_list.append(data_net_exp)
            edge_percent.append(
                np.sum(net_exp['net']) / (net_exp['exp'].shape[0] * net_exp['exp'].shape[0] - net_exp['exp'].shape[0]))
        edge_percent = sum(edge_percent) / len(edge_percent)
        # 将数据批量转换为单个Data对象，并为每个节点和边分配batch值
        if adddata is not None:
            for adddata_l in adddata:
                data_list.extend(adddata_l['data_list'])
                edge_percent = edge_percent + adddata_l['edge_percent']
            edge_percent = edge_percent / (1 + len(adddata))
        batch = Batch.from_data_list(data_list)
        return batch, edge_percent


def cal_metacell(BRCA_exp_filter_saver, Cnum=100, k=20):
    # 转置DataFrame以按列计算邻居
    BRCA_exp_filter_savert = BRCA_exp_filter_saver.transpose()
    # 使用KNN计算每列的前k个邻居
    neigh = NearestNeighbors(n_neighbors=k, metric='minkowski')
    neigh.fit(BRCA_exp_filter_savert)
    # 获取每列的前k个邻居的索引
    K_list = neigh.kneighbors(BRCA_exp_filter_savert, return_distance=False)
    ALL_C_list = list(range(BRCA_exp_filter_saver.shape[1]))
    max_consecutive_updates = Cnum * 2
    S = [None for x in range(Cnum)]
    old_S = copy.deepcopy(S)
    Nc_max_list = np.zeros((1, Cnum))
    counter = 0
    if BRCA_exp_filter_savert.shape[0] <= Cnum:
        print(
            f"The number of cells to be processed ({str(BRCA_exp_filter_savert.shape[0])}) is less than (or equal) the number of Meta-cells ({str(Cnum)})!")
        return BRCA_exp_filter_saver
    while counter < max_consecutive_updates:
        ALL_C_list_current = [x for x in ALL_C_list if x not in S]
        for c in ALL_C_list_current:
            if c not in S:
                Nc_max = len(set(K_list[c]))
                for j in S:
                    if j is not None:
                        Nc = len(set(K_list[c]) | set(K_list[j]))
                        if Nc > Nc_max:
                            Nc_max = Nc
                if np.any(Nc_max > Nc_max_list):
                    S[np.argmin(Nc_max_list)] = c
                    Nc_max_list[0, np.argmin(Nc_max_list)] = Nc_max
                elif Nc_max == (k * 2) and c < np.max(S):
                    S[np.argmax(S)] = c
                    Nc_max_list[0, np.argmax(S)] = Nc_max
                if np.array_equal(S, old_S):
                    counter += 1
                else:
                    old_S = copy.deepcopy(S)
                    counter = 0
        for cn in range(Cnum):
            c = S[cn]
            Nc_max = len(set(K_list[c]))
            for j in S:
                if j is not None and j != c:
                    Nc = len(set(K_list[c]) | set(K_list[j]))
                    if Nc > Nc_max:
                        Nc_max = Nc
            if np.any(Nc_max > Nc_max_list[0, cn]):
                S[cn] = c
                Nc_max_list[0, cn] = Nc_max
    S = np.sort(S)
    assert None not in S, "Meta-cell list contains None!!!"
    assert len(S) == len(set(S)), "Meta-cell list contains duplicate values!!!"
    BRCA_exp_filter_saver = pd.DataFrame()
    for si in range(0, Cnum):
        new_value = (
            BRCA_exp_filter_savert.iloc[K_list[S[si], 0:int(BRCA_exp_filter_savert.shape[0] / Cnum)], :].mean(axis=0))
        BRCA_exp_filter_saver[str('c' + str(si))] = new_value
    return BRCA_exp_filter_saver


def Construct_training_network(BRCA_exp_filter_saver, KEGG, parm, test_pathway=None,
                               Other_Pathway=None, human_network=None, LLM_filepath=None,
                               Adddatabse=None, theta=None, device='cpu'):
    [exp, adj_matrix, new_row] = from_cancer_create(BRCA_exp_filter_saver, KEGG, parm,
                                                    test_pathway=test_pathway,
                                                    Other_Pathway=Other_Pathway,
                                                    human_network=human_network)
    if exp is not None:
        description = f"Pathway: {Other_Pathway}, total contain {exp.shape[0]} genes, and {np.sum(adj_matrix)} links!"

        data_net_exp = matrix2Data(adj_matrix, exp, num=0, adj2data=True, Adddatabse=Adddatabse, theta=theta,
                                   LLM_filepath=LLM_filepath).to(device)
        edge_perc = (np.sum(adj_matrix) / (exp.shape[0] * exp.shape[0] - exp.shape[0]))
    else:
        data_net_exp = None
        edge_perc = None
    return new_row, data_net_exp, edge_perc


def create_batch_dataset_from_cancer(filepath='CancerDatasets/DCA/BRCA_output.csv',
                                     test_pathway='hsa05224',
                                     test=False,
                                     device=None,
                                     metacell=True,
                                     Cnum=100,
                                     k=20,
                                     lim=200,
                                     return_list=False,
                                     Adddatabse=True,
                                     theta=None,
                                     LLM_filepath=None):
    # 1. 读取基因表达数据
    if theta is None:
        theta = {'PMI': 30, 'PPC': 0.3, 'database': 'RegNetwork'}
    if metacell:
        # BRCA_exp_filter_saver = pd.read_csv(filepath.replace("output", "input"))
        BRCA_exp_filter_saver = pd.read_csv(filepath)
        BRCA_exp_filter_saver.set_index(BRCA_exp_filter_saver.columns[0], inplace=True)
        BRCA_exp_filter_saver.drop(columns=BRCA_exp_filter_saver.columns[0], inplace=True)
        BRCA_exp_filter_saver = cal_metacell(BRCA_exp_filter_saver, Cnum=Cnum, k=k)
    else:
        BRCA_exp_filter_saver = pd.read_csv(filepath)
        BRCA_exp_filter_saver.set_index(BRCA_exp_filter_saver.columns[0], inplace=True)
        BRCA_exp_filter_saver.drop(columns=BRCA_exp_filter_saver.columns[0], inplace=True)

    # 2. 读取KEGG pathway信息
    KEGG = load_KEGG()

    # 3. 读取Regnetwork信息
    Regnetwork_path = 'pathway/Regnetwork/2022.human.source'
    dtypes = {1: str, 3: str}
    human_network = pd.read_csv(Regnetwork_path, sep='\t', header=None, dtype=dtypes)
    human_network.columns = ['TF', 'TF_ID', 'Gene', 'Gene_ID']
    # 4. Create exp-net
    columns = {'Pathway': None, 'NUM_ORIG': None, 'NUM_PCC': None, 'NUM_MI': None}
    network_percent = pd.DataFrame(columns=columns)
    if test:
        parm = {'pear_percent': 1, 'MI_percent': 1, 'pmi_percent': 0.001}  # 这里 1 表示 找到的高相关性的边不低于节点数量的  1 倍
        [exp, adj_matrix, _] = from_cancer_create(BRCA_exp_filter_saver, KEGG, parm,
                                                  lim=lim,
                                                  test_pathway=None,
                                                  Other_Pathway=test_pathway,
                                                  human_network=human_network)
        batch = matrix2Data(adj_matrix, exp, num=0, adj2data=True, Adddatabse=Adddatabse, theta=theta,
                            LLM_filepath=LLM_filepath).to(device)
        print((f" Pathway: {test_pathway}, total contain {exp.shape[0]} genes, and {np.sum(adj_matrix)} links!"))
        return batch
    else:
        data_list = []
        edge_percent = []
        pathway_ID_list = KEGG.keys()
        pathway_ID_list = list(pathway_ID_list)
        parm = {'pear_percent': 1, 'MI_percent': 1, 'pmi_percent': 0.001}
        pbar = tqdm(pathway_ID_list, ncols=100)
        for Other_Pathway in pbar:
            [exp, adj_matrix, new_row] = from_cancer_create(BRCA_exp_filter_saver, KEGG, parm,
                                                            test_pathway=test_pathway,
                                                            Other_Pathway=Other_Pathway,
                                                            human_network=human_network)
            network_percent = pd.concat([network_percent, pd.DataFrame([new_row])], ignore_index=True)
            if exp is not None:
                pbar.set_description(
                    f" Pathway: {Other_Pathway}, total contain {exp.shape[0]} genes, and {np.sum(adj_matrix)} links!")
                data_net_exp = matrix2Data(adj_matrix, exp, num=0, adj2data=True, Adddatabse=Adddatabse, theta=theta,
                                           LLM_filepath=LLM_filepath).to(device)
                data_list.append(data_net_exp)
                edge_percent.append(np.sum(adj_matrix) / (exp.shape[0] * exp.shape[0] - exp.shape[0]))

        # Parallel_result_list = Parallel(n_jobs=-1)(
        #     delayed(Construct_training_network)(BRCA_exp_filter_saver, KEGG, parm, test_pathway=test_pathway,
        #                                         Other_Pathway=Other_Pathway, human_network=human_network,
        #                                         LLM_filepath=LLM_filepath,Adddatabse=Adddatabse, theta=theta,
        #                                         device=device) for Other_Pathway in pbar)
        # data_list = []
        # edge_percent = []
        # for new_row, data_net_exp, edge_perc in Parallel_result_list:
        #     if data_net_exp is not None:
        #         data_list.append(data_net_exp)
        #         edge_percent.append(edge_perc)

        # plot_GRN_percent(network_percent)
        if return_list:
            return data_list, edge_percent
        else:
            batch = Batch.from_data_list(data_list)
            edge_percent = sum(edge_percent) / len(edge_percent)
            return batch, edge_percent


if __name__ == '__main__':
    from PCA_CMI import pca_cmi
    import powerlaw
    # celltype = ['T_cell', 'B_cell', 'Myeloid', 'Cancer', 'DC', 'EC', 'Fibroblast', 'Mast']
    # test_pathway = 'hsa05224'
    # Cnum = 100
    # k = 20
    # KEGG = load_KEGG(kegg_file='kegg/KEGG_all_pathway.pkl')
    # Regnetwork_path = 'Regnetwork/2022.human.source'
    # dtypes = {1: str, 3: str}
    # human_network = pd.read_csv(Regnetwork_path, sep='\t', header=None, dtype=dtypes)
    # human_network.columns = ['TF', 'TF_ID', 'Gene', 'Gene_ID']
    # meatacell = False
    # for cell in celltype:
    #     filepath = '/home/wcy/Diff_comple/Cancer_datasets/BRCA/BRCA_Tumor_' + str(cell) + '_output.csv'
    #     BRCA_exp_filter_saver = pd.read_csv(filepath)
    #     BRCA_exp_filter_saver.set_index(BRCA_exp_filter_saver.columns[0], inplace=True)
    #     BRCA_exp_filter_saver.drop(columns=BRCA_exp_filter_saver.columns[0], inplace=True)
    #     BRCA_exp_filter_saver_metacell = cal_metacell(BRCA_exp_filter_saver, Cnum=Cnum, k=k)
    #     filepath = filepath.replace("output", "output_meta")
    #     if BRCA_exp_filter_saver is not None:
    #         parm = {'pear_percent': 1, 'MI_percent': 1, 'pmi_percent': 0.001}
    #         [exp, adj_matrix, new_row] = from_cancer_create(BRCA_exp_filter_saver_metacell,
    #                                                         KEGG,
    #                                                         parm,
    #                                                         test_pathway=None,
    #                                                         Other_Pathway=test_pathway,
    #                                                         human_network=human_network,
    #                                                         TF_file='/home/wcy/Diff_comple/GRN/TF.txt')
    #         data1 = np.sum(adj_matrix, axis=0)
    #         fit1 = powerlaw.Fit(data1)
    #         data2 = np.sum(adj_matrix, axis=1)
    #         fit2 = powerlaw.Fit(data2)
    #         data3 = data1 + data2
    #         fit3 = powerlaw.Fit(data3)
    #         print("Alpha (exponent,0):", fit1.alpha, "Alpha (exponent,1):", fit2.alpha, "Alpha (exponent,sum):",
    #               fit3.alpha)
    #         data = {"net": adj_matrix, "exp": np.array(exp), "genename": exp.index}
    #         print(new_row)
    #
    #         filename = filepath.replace("csv", "data")
    #         f = open(filename, 'wb')
    #         pickle.dump(data, f)
    #         f.close()
    #
    #         sio.savemat(filepath.replace("csv", "mat"),
    #                     {"net": adj_matrix, "exp": np.array(exp), "genename": exp.index})
    #         print(filename + ' is OK!!!!')
    #

    celltype = ['Tcell', 'Bcell', 'NK', 'Mye', 'HSC', 'Tumor']
    tissue = ['Tumor', 'Adjacent_liver']
    test_pathway = 'hsa05225'
    Cnum = 100
    k = 20
    KEGG = load_KEGG(kegg_file='kegg/KEGG_all_pathway.pkl')
    Regnetwork_path = 'Regnetwork/2022.human.source'
    dtypes = {1: str, 3: str}
    human_network = pd.read_csv(Regnetwork_path, sep='\t', header=None, dtype=dtypes)
    human_network.columns = ['TF', 'TF_ID', 'Gene', 'Gene_ID']
    meatacell = False
    for tis in tissue:
        for cell in celltype:
            if os.path.exists('/home/wcy/Diff_comple/Cancer_datasets/HCC/preprocess/HCC_' + str(tis) + '_' + str(cell) + '_input.csv'):
                filepath = '/home/wcy/Diff_comple/Cancer_datasets/HCC/preprocess/HCC_' + str(tis) + '_' + str(cell) + '_input.csv'
                BRCA_exp_filter_saver = pd.read_csv(filepath)
                BRCA_exp_filter_saver.set_index(BRCA_exp_filter_saver.columns[0], inplace=True)
                BRCA_exp_filter_saver.drop(columns=BRCA_exp_filter_saver.columns[0], inplace=True)
                BRCA_exp_filter_saver_metacell = cal_metacell(BRCA_exp_filter_saver, Cnum=Cnum, k=k)
                filepath = filepath.replace("input", "input_meta")
                if BRCA_exp_filter_saver is not None:
                    parm = {'pear_percent': 1, 'MI_percent': 1, 'pmi_percent': 0.001}
                    [exp, adj_matrix, new_row] = from_cancer_create(BRCA_exp_filter_saver_metacell,
                                                                    KEGG,
                                                                    parm,
                                                                    test_pathway=None,
                                                                    Other_Pathway=test_pathway,
                                                                    human_network=human_network,
                                                                    TF_file='/home/wcy/Diff_comple/GRN/TF.txt')
                    data1 = np.sum(adj_matrix, axis=0)
                    fit1 = powerlaw.Fit(data1)
                    data2 = np.sum(adj_matrix, axis=1)
                    fit2 = powerlaw.Fit(data2)
                    data3 = data1 + data2
                    fit3 = powerlaw.Fit(data3)
                    print("Alpha (exponent,0):", fit1.alpha, "Alpha (exponent,1):", fit2.alpha, "Alpha (exponent,sum):",
                          fit3.alpha)
                    data = {"net": adj_matrix, "exp": np.array(exp), "genename": exp.index}
                    print(new_row)

                    filename = filepath.replace("csv", "data")
                    f = open(filename, 'wb')
                    pickle.dump(data, f)
                    f.close()

                    sio.savemat(filepath.replace("csv", "mat"),
                                {"net": adj_matrix, "exp": np.array(exp), "genename": exp.index})
                    print(filename + ' is OK!!!!')
else:
    from pathway.PCA_CMI import pca_cmi
    from pathway.PCA_PCC import pca_pcc
