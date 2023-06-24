import networkx as nx
from collections import Counter
from utils import *
from utils_super_nodes import *


# 分析得到的结果社区性质
def analysis(result_classes, orig, label):
    print("原始图：", orig)
    community_list, label_community_list = \
        get_community_from_result_classes(result_classes, [v for v in orig.nodes()], label)

    all_nodes = []
    all_edges = []
    interval_edges = []

    print("输出预测和真实情况...")
    count_label = Counter(label)
    count_pre = Counter(result_classes)
    print(count_label)
    print(count_pre)

    print("社区内情况........................................")
    for i in range(len(community_list)):
        ori_acc = 0
        all_nodes += community_list[i]
        ori_sub_g = nx.subgraph(orig, community_list[i])
        all_edges += [e for e in ori_sub_g.edges()]

        for j in ori_sub_g.edges():
            if label_community_list[i][community_list[i].index(j[0])] == label_community_list[i][community_list[i].index(j[1])]:
                ori_acc += 1
        count = Counter(label_community_list[i])

        # 每个社区的节点数和节点编号，可以先不输出
        # print(len(community_list[i]), community_list[i])

        print(ori_sub_g, '社区'+str(i))
        print("真实标签分布：", count)
        print("正常边：", ori_acc,
              "错误边：", len(ori_sub_g.edges())-ori_acc,
              "总边数：", len(ori_sub_g.edges()),
              "连通分量数目：", nx.number_connected_components(ori_sub_g),
              "节点数", len(community_list[i]))
        print('-------------------------------------------------------------------------')

    all_nodes_g = nx.subgraph(orig, all_nodes)
    for i in all_nodes_g.edges():
        if (i[0], i[1]) in all_edges or (i[1], i[0]) in all_edges:
            continue
        else:
            interval_edges.append(i)
    interval_edges_acc = 0
    err_edges = []
    for i in interval_edges:
        if label[i[0]] == label[i[1]]:
            interval_edges_acc += 1
        else:
            err_edges.append(i)

    print("社区之间情况........................................：",
          "正常边：", interval_edges_acc,
          "异常边：", len(interval_edges)-interval_edges_acc,
          "总边数：", len(interval_edges))


# 分析加边和减边的性质
def analysis_add_and_delete(add_list, delete_list, label):
    lap = [e for e in add_list if (e[0], e[1]) in delete_list or (e[1], e[0]) in delete_list]
    print("lap", lap)

    # for i in add_list:
    #     print(i, )

    # for i in delete_list:
    #     print(i, label[i[0]], label[i[1]])


# 分析超级节点内部性质
# 连通分量
def analysis_super_nodes(nodes, g, label):
    sub_g = nx.subgraph(g, nodes)
    componets = nx.connected_components(sub_g)
    for i in componets:
        print(i, [label[v] for v in i])


import pandas as pd
import networkx.algorithms.community as nx_comm
def analysis_super_nodes_big(result_classes, h, g, label):
    community_list, label_community_list = \
        get_community_from_result_classes(result_classes, [v for v in g.nodes()], label)
    h_community = []
    classes_embed_list = []
    for i in community_list:
        h_community.append(h[i])
        classes_embed_list.append(h[i].mean(dim=0))
    #  计算每个类别中的分布
    for i in range(len(h_community)):
        # cos = get_cos_sim(h_community[i])
        # plot_classes_sim(cos, str(i))
        m = nx.average_clustering(nx.Graph(nx.subgraph(g, community_list[i])))
        print(str(i), m)
        # l = []
        # for u in cos.tolist():
        #     l += u
        # se1 = pd.cut(l, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # print(str(i))
        # for s in se1.value_counts():
        #     print(s/(len(community_list[i])*len(community_list[i])))
    #  计算总体分布
    # classes_embed_tensor = torch.stack(classes_embed_list)
    # cos_classes_sim = get_cos_sim_k(classes_embed_tensor, len(community_list))
    # print(cos_classes_sim)

    # # 计算每个类别排名的边
    # for i in range(len(h_community)):
    #     community_classes = community_list[i]
    #
    #     sub_g = nx.subgraph(g, community_classes)
    #
    #     # 获取社区内部所有节点对
    #     # pro = product(community_classes, community_classes)
    #     pro = [e for e in sub_g.edges()]
    #     edge = []
    #     edges = []
    #     print("计算所有节点对........")
    #     # for m in pro:
    #     #     if m[0] < m[1]:
    #     #         edge.append(m)
    #     #         # 映射关系
    #     #         edges.append((community_classes.index(m[0]), community_classes.index(m[1])))
    #
    #     for m in pro:
    #         edge.append(m)
    #         # 映射关系
    #         edges.append((community_classes.index(m[0]), community_classes.index(m[1])))
    #     cos_distance = get_cos_sim(h_community[i])
    #
    #     add_cos_distance = cos_distance[change_list(edges)]
    #
    #     print("排序.......")
    #     zip_list = zip(edge, add_cos_distance)
    #     temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
    #     result = zip(*temp)
    #     new_edge, new_add_cos_distance = [list(x) for x in result]
    #
    #     print("写入文件......")
    #     label_classes = []
    #     for k in range(len(new_edge)):
    #         label_classes.append(((new_edge[k][0], new_edge[k][1]),
    #                               (label[new_edge[k][0]], label[new_edge[k][1]]),
    #                               new_add_cos_distance[k]))
    #     writetxt3(label_classes, str(i))


from data import *
if __name__ == '__main__':
    a, g, x, edge_index, label = prepare_data_cora()
    nodes = \
        [102, 114, 115, 138, 234, 317, 359, 416, 506, 519, 603, 610, 716, 742, 784, 795, 826, 840, 902, 959, 960, 1077, 1248, 1283, 1352, 1412, 1483, 1561, 1572, 1620, 1743, 1768, 1776, 1837, 1852, 1878, 1918, 2078, 2097, 2098, 2256, 2287, 2288, 2324, 2450, 2481, 2506, 2529, 2591, 2650, 2664]

    analysis_super_nodes(nodes, g, label)

