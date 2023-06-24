import dgl
import numpy as np
from dgl.data.citation_graph import CiteseerGraphDataset, PubmedGraphDataset, CoraGraphDataset
import networkx as nx
import torch
from utils import *
import scipy.sparse as sp
import sklearn.preprocessing as preprocess


device = torch.device("cuda")


# 根据数据集名称返回相应数据
def prepare_data_all(prog_args):
    if prog_args.data == 'cora':
        return prepare_data_cora()
    elif prog_args.data == 'karate':
        return prepare_data()
    elif prog_args.data == 'citeseer':
        return prepare_data_citeseer()
    elif prog_args.data == 'pubmed':
        return prepare_data_pubmed()
    else:
        print('数据名称错误')
        exit(0)


# 根据边生成图g和邻接矩阵a
# 绘制图g
def prepare_data():
    '''
    所有78条边都存储在两个numpy数组中, 一个用于源端点而另一个用于目标端点
    '''
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
                    10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
                    25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
                    32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
                    33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
                    5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
                    24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
                    29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
                    31, 32])
    # 边缘在DGL中是有方向的； 使它们双向
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # 构建图
    g = dgl.graph((u, v))

    # 没有特征矩阵X，初始化为对角矩阵
    x = np.identity(g.number_of_nodes())
    x = torch.tensor(x).type(torch.FloatTensor).to(device)

    g = g.to_networkx().to_undirected()

    # 获取邻接矩阵
    a = torch.Tensor(nx.adjacency_matrix(g).todense(),).type(torch.FloatTensor)
    a = a.to(device)

    label = []

    G = nx.karate_club_graph()

    for i in range(34):
        if G.nodes[i]["club"] == 'Mr. Hi':
            label.append(0)
        else:
            label.append(1)

    return a, g, x, get_edge_index(g), label


# 加载cora数据集
def prepare_data_cora():
    dataset = CoraGraphDataset()
    print('Number of categories:', dataset.num_classes)
    g = dataset[0]
    print(g)

    # 特征矩阵
    x = g.ndata['feat'].type(torch.FloatTensor).to(device)

    # label可用于对比
    label = g.ndata['label']
    label = label.tolist()

    # 转化为无向图
    g = g.to_networkx().to_undirected()

    # 获取邻接矩阵
    a = torch.Tensor(nx.adjacency_matrix(g).todense()).type(torch.FloatTensor)
    a = a.to(device)

    return a, g, x, get_edge_index(g), label


# 加载Citeseer数据集
def prepare_data_citeseer():
    dataset = CiteseerGraphDataset()
    print('Number of categories:', dataset.num_classes)
    g = dataset[0]
    print(g)

    # 特征矩阵
    x = g.ndata['feat'].type(torch.FloatTensor).to(device)

    # label可用于对比
    label = g.ndata['label']
    label = label.tolist()

    # 转化为无向图
    g = g.to_networkx().to_undirected()

    # 获取邻接矩阵
    a = torch.Tensor(nx.adjacency_matrix(g).todense()).type(torch.FloatTensor)
    a = a.to(device)

    return a, g, x, get_edge_index(g), label


# 加载pubmedr数据集
def prepare_data_pubmed():
    dataset = PubmedGraphDataset()
    print('Number of categories:', dataset.num_classes)
    g = dataset[0]
    print(g)

    # 特征矩阵
    x = g.ndata['feat'].type(torch.FloatTensor).to(device)

    # label可用于对比
    label = g.ndata['label']
    label = label.tolist()

    # 转化为无向图
    g = g.to_networkx().to_undirected()

    # 获取邻接矩阵
    a = torch.Tensor(nx.adjacency_matrix(g).todense()).type(torch.FloatTensor)
    a = a.to(device)

    return a, g, x, get_edge_index(g), label


import pandas as pd
# 加载wiki数据集
def prepare_data_wiki():
    f = open('./dataset/wiki/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()

    g = nx.MultiGraph()
    g.add_edges_from(adj)
    g = g.to_undirected()

    f = open('./dataset/wiki/wiki_category.txt', 'r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('./dataset/wiki/tfidf.txt', 'r')
    fea_idx = []
    fea = []
    adj = np.array(adj)
    adj = np.vstack((adj, adj[:, [1, 0]]))
    adj = np.unique(adj, axis=0)

    labelset = np.unique(label)
    labeldict = dict(zip(labelset, range(len(labelset))))
    label = np.array([labeldict[x] for x in label])
    adj = sp.csr_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(label), len(label)))

    for line in f.readlines():
        line = line.split()
        fea_idx.append([int(line[0]), int(line[1])])
        fea.append(float(line[2]))
    f.close()

    fea_idx = np.array(fea_idx)
    features = sp.csr_matrix((fea, (fea_idx[:, 0], fea_idx[:, 1])), shape=(len(label), 4973)).toarray()
    scaler = preprocess.MinMaxScaler()
    # features = preprocess.normalize(features, norm='l2')
    features = scaler.fit_transform(features)
    features = torch.FloatTensor(features)

    # 获取邻接矩阵
    a = torch.Tensor(nx.adjacency_matrix(g).todense()).type(torch.FloatTensor)
    a = a.to(device)

    x = features.type(torch.FloatTensor).to(device)

    label = label.tolist()

    return a, g, x, get_edge_index(g), label


if __name__ == '__main__':
    prepare_data_wiki()
