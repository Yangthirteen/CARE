from data import *
from itertools import product


# 分析0号社区的结构
# 查找结构洞的几个部分
def analysis_community_k(g, k, label):
    community_nodes = []
    community_nodes_degree = []
    community_edges = []

    for v in g.nodes():
        if label[v] == k:
            community_nodes.append(v)

    print("节点", len(community_nodes), community_nodes)
    # 按节点度排序，从节点度较大的开始画
    sub_g_k = g.subgraph(community_nodes)
    for v in community_nodes:
        community_nodes_degree.append(sub_g_k.degree(v))

    for e in sub_g_k.edges():
        community_edges.append(e)
    print("边", len(community_edges), community_edges)

    print("度", len(community_nodes_degree), community_nodes_degree)


# 参数：最后结果result_classes，加边列表
# 根据result_classes找到0号社区的节点数较多的两部分
# 根据加边列表找出哪些加边在结构洞中
def compare_add_list_in_structure_hole(result_classes, label, label_k, h_list):
    # 加边列表被保存为文件txt
    add_list = readtxt("pic/add_list.txt")
    # 根据结果获取两部分0号社区
    community_list, label_community_list = \
        get_community_from_result_classes(result_classes, [v for v in range(len(result_classes))], label)
    community_zero = []
    for i in range(len(community_list)):
        zero_nodes = []
        for j in community_list[i]:
            if label[j] == label_k:
                zero_nodes.append(j)
        community_zero.append(zero_nodes)

    two_part_zero = []
    for k in range(2):
        maxList = max(community_zero, key=lambda com: len(com))
        community_zero.remove(maxList)
        two_part_zero.append(maxList)

    edge_nodes = list(product(two_part_zero[0], two_part_zero[1]))

    structure_edges = []
    for e in add_list:
        if e in edge_nodes:
            structure_edges.append(e)

    print("在"+str(label_k)+"号社区两部分中的加边数量为：", len(structure_edges))

    # writetxt(structure_edges, "new_add_list"+str(label_k))

    # if label_k == 0:
    #     analysis_sim_in_structure_hole(h_list[0], edge_nodes[0], edge_nodes[1])


# 查看0号社区结构洞的边在线性层得到的H的相似度
def analysis_sim_in_structure_hole(h, part_one, part_two):
    edge_nodes = list(product(part_one, part_two))
    index_cos, index = change_index(edge_nodes)
    cos = get_cos_sim(h)
    print("结构洞中边的相似度在线性层得到的H中的相似度分布......")
    print(cos[index_cos])


if __name__ == '__main__':
    a, g, x, edge_index, label = prepare_data_cora()
    analysis_community_k(g, 0, label)


