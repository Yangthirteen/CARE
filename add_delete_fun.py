from add_edges import *
from delete_edges import *


# 删除增加边的总函数
# 在此处设定删除/增加的比例以及路径长度
def add_and_delete(h, g, label, type, add_number, delete_number,
                   result_classes, orig, x, super_result_classes, layer):

    # super_nodes_number = 200-1*layer
    # _, _, _, _, super_nodes_number = read_parameter()
    # super_nodes_number -= 2
    # write_parameter(-1, -1, -1, -1, super_nodes_number)

    res = None
    # print("超级节点结果分析.........")
    # super_nodes_number = 200
    # super_result_classes = classify_spectral(h, super_nodes_number)
    # # analysis(super_result_classes, g, label)
    #
    # analysis_super_nodes_2(h, label, orig, super_result_classes, super_nodes_number)

    # analysis_err_edges_super_nodes(g, h, super_result_classes, label)

    if type == 'delete' or type == 'add+delete':
        # 欧式距离
        # delete_list = delete_edges_eu(h, g, delete_number)
        # 余弦相似度
        # delete_list = delete_edges_cos(h, g, delete_number)
        # sigmoid
        # delete_list = delete_edges_sigmoid(h, g, delete_number)

        # 超级节点之间减边
        # delete_list = delete_edges_super_nodes_cos_dynamic(g, h, super_result_classes, layer)

        # 超级节点内减边
        # delete_list = delete_edges_community_nodes_pair_cos(h, g, super_result_classes)

        # 删除超级节点间所有边
        # delete_list = delete_edges_super_nodes_all(g, h, super_result_classes)

        delete_list = delete_edges_classes(g, h, result_classes, 100, 0.6, label)
    else:
        delete_list = []

    if (type == 'add' or type == 'add+delete'):
        # 路径为k的节点对之间增加边
        # path_k = 3
        # add_list = add_edges(h, g, path_k, add_number, label)

        # 连通分量无连边节点对增加边
        # if nx.number_connected_components(g) <= 1:
        #     print("该图连通分量数量为：1，无法再连通分量之间加边")
        #     exit(-1)

        # 欧式距离
        # add_list = add_edges_componets_eu(h, g, add_number)
        # 余弦相似度
        # add_list = add_edges_componets_cos(h, g, add_number)
        # sigmoid
        # add_list = add_edges_componets_sigmoid(h, g, add_number)

        # 所有无连边节点对增加边

        # 欧式距离
        # add_list = add_edges_nodes_pair_eu(h, g, add_number)
        # 余弦相似度
        # add_list = add_edges_nodes_pair_cos(h, g, add_number)
        # sigmoid
        # add_list = add_edges_nodes_pair_sigmoid(h, g, add_number)

        # 上一轮划分出的社区间加边（编写代码）
        # add_list = add_edges_community_nodes_pair_cos(h, g, result_classes)

        # 超级节点之间加边
        # add_list = add_edges_super_nodes_cos_dynamic(g, h, super_result_classes, x, label, layer)

        # 超级节点内部随机加边（聚集系数大）
        # add_list = add_edges_classes(g, h, result_classes, len(delete_list), 200, label)

        add_list = add_edges_classes_componets(g, h, label)

        # 统计加边的分布
        # res = statistics_add_list_community(add_list, label)
    else:
        add_list = []

    new_g = g.copy()
    if type == 'add':
        new_g.add_edges_from(add_list)
    elif type == 'delete':
        new_g.remove_edges_from(delete_list)
    else:
        new_g.add_edges_from(add_list)
        new_g.remove_edges_from(delete_list)

    result_txt = val_acc(add_list, delete_list, label)

    new_edge_index = get_edge_index(new_g)

    return new_g, new_edge_index, add_list, delete_list, result_txt, res

