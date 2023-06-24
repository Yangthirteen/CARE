import torch
from networkx import *
import torch.nn as nn
from sklearn.decomposition import PCA, KernelPCA  # 加载PCA算法包
import networkx as nx
from itertools import product
import math
import random
from plot_result import *


# 打印删减边的结果
def print_result(result_total):
    for i in result_total:
        print(i)


# 根据图得到边列表
# [source_list, dist_list]
def get_edge_index(g):
    edges = g.edges()
    list_source = []
    list_target = []
    for i in edges:
        list_source.append(i[0])
        list_target.append(i[1])
    edge_index = torch.tensor([list_source, list_target]).type(torch.LongTensor).cuda()
    return edge_index


# 获取嵌入的欧氏距离矩阵
def get_euclidean_distance(h):
    pca = PCA(n_components=8)
    reduced_x = pca.fit_transform(h.cpu().detach().numpy())
    reduced_x = torch.Tensor(reduced_x)

    pdist = nn.PairwiseDistance(p=2)
    euclidean_distance = pdist(reduced_x.unsqueeze(1), reduced_x.unsqueeze(0))
    return euclidean_distance


# 计算余弦相似度矩阵
def get_cos_sim_ori(h):
    pca = PCA(n_components=8)
    reduced_x = pca.fit_transform(h.cpu().detach().numpy())
    reduced_x = torch.Tensor(reduced_x)

    cos_sim = torch.cosine_similarity(reduced_x.unsqueeze(1), reduced_x.unsqueeze(0), dim=-1)

    return cos_sim


# 计算余弦相似度矩阵
# 归一化
def get_cos_sim(h):
    pca = PCA(n_components=8)
    reduced_x = pca.fit_transform(h.cpu().detach().numpy())
    reduced_x = torch.Tensor(reduced_x)

    zmax = reduced_x.max(dim=1, keepdim=True)[0]
    zmin = reduced_x.min(dim=1, keepdim=True)[0]
    z_std = (reduced_x - zmin) / (zmax - zmin)

    cos_sim = torch.cosine_similarity(z_std.unsqueeze(1), z_std.unsqueeze(0), dim=-1)
    # torch.set_printoptions(profile="full")
    # print(cos_sim)
    return cos_sim

# 计算余弦相似度矩阵
# 归一化
def get_cos_sim_k(h, k):
    pca = PCA(n_components=k)
    reduced_x = pca.fit_transform(h.cpu().detach().numpy())
    reduced_x = torch.Tensor(reduced_x)

    zmax = reduced_x.max(dim=1, keepdim=True)[0]
    zmin = reduced_x.min(dim=1, keepdim=True)[0]
    z_std = (reduced_x - zmin) / (zmax - zmin)

    cos_sim = torch.cosine_similarity(z_std.unsqueeze(1), z_std.unsqueeze(0), dim=-1)
    # torch.set_printoptions(profile="full")
    # print(cos_sim)
    return cos_sim


# 改变节点对形式，主要是为了方便从欧氏距离矩阵中获取相应节点对的元素
# [(),(),(),...,()]->[[], []]
def change_list(edge_list):
    list_first = []
    list_second = []
    for i in edge_list:
        list_first.append(i[0])
        list_second.append(i[1])

    return [list_first, list_second]


def change_list_2(edge_list, E):
    list_first = []
    list_second = []
    for i in edge_list:
        list_first.append(i[0])
        list_second.append(i[1])
    list_first += E[0]
    list_second += E[1]

    return [list_first, list_second]


# [[], []]->[(),(),(),...,()]
def change_list_revers(edge_index):
    total_list = []
    for i in range(len(edge_index[0])):
        if edge_index[0][i] < edge_index[1][i]:
            total_list.append((int(edge_index[0][i]), int(edge_index[1][i])))
        else:
            total_list.append((int(edge_index[1][i]), int(edge_index[0][i])))

    return total_list


# 返回两个结果---主要为了去重
# 用以sigmoid删减边中
def change_index(edge_list):
    list_first = []
    list_second = []
    new_edge_list = []
    for i in edge_list:
        if i[0] < i[1]:
            list_first.append(i[0])
            list_second.append(i[1])
            new_edge_list.append(i)

    return [list_first, list_second], new_edge_list


# 计算每个连通分量中的节点数量
def get_number_of_nodes_of_componets(g):
    componets = nx.connected_components(g)
    number_of_nodes = []
    for i in componets:
        number_of_nodes.append(len(i))
    return number_of_nodes


# 找出所有连通分量之间的不存在路径的节点对
def get_all_no_edges_nodes_component(g):
    componets = nx.connected_components(g)
    no_edges_nodes = []
    componets_lists = []
    for i in componets:
        componets_lists.append(list(i))
    for i in range(len(componets_lists)):
        j = i+1
        while j < len(componets_lists):
            no_edges_nodes += product(componets_lists[i], componets_lists[j])
            j = j+1
    print("节点对数量：", len(no_edges_nodes))

    return no_edges_nodes


# 获取没有连边的节点对，不重复
def get_all_no_edges_nodes(g):
    no_edge_nodes = []
    a = torch.Tensor(nx.adjacency_matrix(g).todense())
    n = g.number_of_nodes()
    for i in range(n):
        j = i+1
        while j < n:
            if a[i][j] == 0:
                no_edge_nodes.append((i, j))
            j = j+1
    print("节点对数量：", len(no_edge_nodes))

    return no_edge_nodes


# 找出路径长度为k的所有节点对
def find_path_length_k(g, k):
    source_target_nodes = []
    for v in g.nodes():
        dict = single_source_shortest_path_length(g, v)
        keys = dict.keys()
        values = dict.values()
        # 限定节点对的顺序只能是（小节点， 大节点），防止出现重复的节点对
        source_target_nodes += [(v, i) for i in keys if dict.get(i) == k and v < i]

    return source_target_nodes


# 验证增加边和删除边的准确率
def val_acc(add_list, delete_list, label):
    add_acc = 0
    for i in add_list:
        if label[i[0]] == label[i[1]]:
            add_acc += 1
        # else:
        #     print(i, label[i[0]], label[i[1]])
    print("---------------------------------")
    delete_acc = 0
    for i in delete_list:
        if label[i[0]] != label[i[1]]:
            delete_acc += 1
            # print(i, label[i[0]], label[i[1]])

    print("增加边准确率：", add_acc, len(add_list))
    print("删除边准确率：", delete_acc, len(delete_list))

    if len(add_list) == 0:
        result_txt = '此轮不增加边---' + \
                     "删除边准确率：" + str(delete_acc) + '_' + str(len(delete_list))
    elif len(delete_list) == 0:
        result_txt = "增加边准确率：" + str(add_acc) + '_' + str(len(add_list)) + '---' + \
                     '此轮不删除边'
    else:
        result_txt = "增加边准确率：" + str(add_acc) + '_' + str(len(add_list)) + '---' + \
                     "删除边准确率：" + str(delete_acc) + '_' + str(len(delete_list))

    return result_txt


# 节点度大的节点增加权重
def create_new_h(h, g, alpha, beta):
    h_t = h.clone()
    z = h.sum()
    for v in g.nodes():
        d_i = g.degree(v)
        h_j = beta*h[v]
        for neighbor in g.neighbors(v):
            d_j = g.degree(neighbor)
            h_j += math.pow(d_j * d_i, alpha)/z*h[neighbor]
        h_t[v] = h_j
    return h_t


# 按照聚类的标签编号顺序选出相应的社区
def get_community_from_result_classes(result_classes, community, label):
    community_list = []
    label_community_list = []
    max_class = max(result_classes)

    for k in range(max_class+1):
        each_community = []
        for i in range(len(result_classes)):
            if result_classes[i] == k:
                each_community.append(community[i])
        community_list.append(each_community)

    for i in community_list:
        label_community = []
        for j in i:
            label_community.append(label[j])
        label_community_list.append(label_community)

    return community_list, label_community_list


# 按照聚类的标签编号顺序选出相应的社区
def result_to_community_order(result_classes, community):
    total_community_list = []
    max_class = max(result_classes)
    for k in range(max_class+1):
        each_community = []
        for i in range(len(result_classes)):
            if result_classes[i] == k:
                each_community.append(community[i])
        total_community_list.append(each_community)
    return total_community_list

# ----------------------------------------------------------------------
# 超级节点——通用
# 计算超级节点的嵌入
def get_super_nodes_embed(h, super_nodes_list):
    super_nodes_embed_list = []
    for i in super_nodes_list:
        temp = h[i].mean(dim=0)
        super_nodes_embed_list.append(temp)
    return super_nodes_embed_list


# ------------------------------------------------------------------------
# 超级节点——加边
# 选取余弦相似度高的超级节点对
def get_super_nodes_pair_cos(cos_super_nodes_sim, k_super_nodes):
    cos = cos_super_nodes_sim

    index1 = torch.nonzero(k_super_nodes[0] <= cos).tolist()
    index2 = torch.nonzero(cos <= k_super_nodes[1]).tolist()
    index = [v for v in index1 if v in index2]

    print("加边所有超级节点相似度.....")
    cos_list = []
    for i in range(cos.shape[0]):
        cos_list.append((i, cos[i]))
        # print(i, cos[i])
    writetxt(cos_list, "cos")

    # 去除重复节点对和自身节点对
    index_cos, index = change_index(index)

    # print("加边超级节点之间的相似度列表：", cos[index_cos])

    return index


# 在两个超级节点之间根据阈值加边
# 笛卡尔积
def add_edges_between_super_nodes(g, super_nodes_one, super_nodes_two, k, cos_sim):
    add_list = []
    edge_nodes = list(product(super_nodes_one, super_nodes_two))
    for i in edge_nodes:
        if g.has_edge(i[0], i[1]) is False and \
                cos_sim[i[0]][i[1]] >= k:
            add_list.append(i)
    return add_list


# 在两个超级节点之间根据阈值加边
# 笛卡尔积
# 增加加边限制：每个节点最多加m条边
def add_edges_between_super_nodes_limit(g, super_nodes_one, super_nodes_two, k, cos_sim, num_add_list):
    m = 5
    add_list = []
    edge_nodes = list(product(super_nodes_one, super_nodes_two))
    for i in edge_nodes:
        if g.has_edge(i[0], i[1]) is False and \
                cos_sim[i[0]][i[1]] >= k and \
                num_add_list[i[0]] < m and num_add_list[i[1]] < m:
            add_list.append(i)
            num_add_list[i[0]] += 1
            num_add_list[i[1]] += 1
    return add_list, num_add_list


# ------------------------------------------------------------------------
# 超级节点——减边
# 选取余弦相似度低的超级节点对
def get_super_nodes_pair_cos_delete(super_nodes_embed_list, k_super_nodes):
    super_nodes_embed_tensor = torch.stack(super_nodes_embed_list)
    cos = get_cos_sim(super_nodes_embed_tensor)
    index1 = torch.nonzero(k_super_nodes[0] <= cos).tolist()
    index2 = torch.nonzero(cos <= k_super_nodes[1]).tolist()
    index = [v for v in index1 if v in index2]

    # 去除重复节点对和自身节点对
    index_cos, index = change_index(index)

    print("减边：超级节点之间的相似度列表：", cos[index_cos])

    return index


# 在两个超级节点之间根据阈值减边
# 笛卡尔积
def delete_edges_between_super_nodes(g, super_nodes_one, super_nodes_two, k, cos_sim):
    delete_list = []
    edge_nodes = list(product(super_nodes_one, super_nodes_two))
    for i in edge_nodes:
        if g.has_edge(i[0], i[1]) is True and \
                cos_sim[i[0]][i[1]] <= k:
            delete_list.append(i)
    return delete_list


# -------------------------------------------------------------------------------
# 统计工具方法


# 统计20w条边中正确边分布情况
def statistics_add_list(add_list, label):
    period = 10
    length_each = int(len(add_list)/period)
    res = []
    for i in range(period):
        right_edges = 0
        offset = i*length_each
        for j in range(length_each):
            if label[add_list[j+offset][0]] == label[add_list[j+offset][1]]:
                right_edges += 1
        acc = right_edges/length_each
        res.append(acc)
    print("统计加边的分布.....")
    print(res)
    return res


# 统计20w条边中正确边分布情况(仅统计0(node_k)号节点结构洞总边的分布)
def statistics_add_list_for_node_k(add_list, label, node_k):
    period = 10
    length_each = int(len(add_list)/period)
    res = []
    for i in range(period):
        right_edges = 0
        offset = i*length_each
        for j in range(length_each):
            if label[add_list[j+offset][0]] == label[add_list[j+offset][1]] \
                    and label[add_list[j+offset][0]] == node_k:
                right_edges += 1
        acc = right_edges/length_each
        res.append(acc)
    print("统计加边的分布.....")
    print(res)
    return res


# 统计20w条边中正确边分布情况(正确加边中各个社区分布)
def statistics_add_list_community(add_list, label):
    count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    add_list_zero = []
    for i in range(len(add_list)):
        if label[add_list[i][0]] == label[add_list[i][1]]:
            count_dict[label[add_list[i][0]]] += 1
            if label[add_list[i][0]] == 0:
                add_list_zero.append(add_list[i])
    print("统计加边的分布.....")
    print(count_dict)

    writetxt(add_list_zero, "new_add_list" + str(0))

    return count_dict


# 将加边写入文件保存
# 写入文件
def writetxt(data, name):
    fileName = 'pic/'+name+'.txt'
    with open(fileName, 'w') as file:
        for i in data:
            file.write(str(i[0])+','+str(i[1])+'\n')
    file.close()


# 写入文件
# 写入文件
def writetxt2(data, name):
    fileName = 'pic/'+name+'.txt'
    k = 0
    with open(fileName, 'w') as file:
        for i in data:
            file.write(str(k)+"---"+'\n')
            m = 0
            for j in i:
                file.write("("+str(m) + "-" + str(j)+")")
                m += 1
            file.write('\n')
            k += 1
    file.close()


def writetxt3(data, name):
    fileName = 'pic/'+name+'.txt'
    with open(fileName, 'w') as file:
        for i in data:
            file.write(str(i)+'\n')
    file.close()


# 读取文件list
def readtxt(file_name):
    add_list = []
    file = open(file_name, 'r', encoding='utf-8')
    for line in file.readlines():
        line = line.strip().split(',')
        if len(line) > 0:
            if int(line[0]) < int(line[1]):
                add_list.append((int(line[0]), int(line[1])))
            else:
                add_list.append((int(line[1]), int(line[0])))
    return add_list

# ------------------------
# 正确的加边，并分析边的情况
#增加同一类中的边——---填补结构洞
def add_0(g, label):
    label_k = [0, 5]
    num = [300, 200]
    ori_g = g.copy()
    for k in range(len(label_k)):
        l = []
        add_list = []

        for i in g.nodes():
            if label[i] == label_k[k]:
                l.append(i)

        while len(add_list) < num[k]:
            idx0 = random.randrange(0, len(l))
            idx1 = random.randrange(0, len(l))
            if idx0 == idx1 or (l[idx0], l[idx1]) in add_list:
                continue
            add_list.append((l[idx0], l[idx1]))
        g.add_edges_from(add_list)
        analysis_correct_edges(add_list, ori_g)

    return g


def analysis_correct_edges(add_list, g):
    for i in add_list:
        if nx.has_path(g, i[0], i[1]):
            print("路径长度", nx.shortest_path_length(g, i[0], i[1]),
                "节点度", g.degree(i[0]), g.degree(i[1]),
                "节点编号", i[0], i[1])
        else:
            print("路径长度", "无路径",
                "节点度", g.degree(i[0]), g.degree(i[1]),
                "节点编号", i[0], i[1])


# 将图处理为连通图
def add_edges_classes_componets(g, h, label):
    cos_distance = get_cos_sim(h)
    add_list = []
    print(nx.number_connected_components(g))
    for c in nx.connected_components(g):
        edges = list(product(list(c), [v for v in g.nodes if v not in list(c)]))
        cos_edges = list(cos_distance[change_list(edges)])

        zip_list = zip(edges, cos_edges)
        temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
        result = zip(*temp)
        new_add_list, new_add_cos_distance = [list(x) for x in result]

        add_list += new_add_list[:1]

    return add_list
