import networkx as nx

from classify_fun import *
from analysis_result import *
from utils_super_nodes import *


# 删除边——欧式距离最大
def delete_edges_eu(h, g, delete_number):
    delete_list = find_path_length_k(g, 1)
    euclidean_distance = get_euclidean_distance(h)
    delete_euclidean_distance = euclidean_distance[change_list(delete_list)]

    zip_list = zip(delete_list, delete_euclidean_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
    result = zip(*temp)
    new_delete_list, new_delete_euclidean_distance = [list(x) for x in result]

    new_delete_list = new_delete_list[:delete_number]

    return new_delete_list


# 删除边——余弦相似度最小
def delete_edges_cos(h, g, delete_number):
    delete_list = find_path_length_k(g, 1)
    cos_distance = get_cos_sim(h)
    delete_cos_distance = cos_distance[change_list(delete_list)]

    zip_list = zip(delete_list, delete_cos_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=False)
    result = zip(*temp)
    new_delete_list, new_delete_euclidean_distance = [list(x) for x in result]

    # new_delete_list = new_delete_list[:delete_number]

    new_delete_list = select_cos_delete_k(new_delete_list, new_delete_euclidean_distance)

    return new_delete_list


# 删除边——sigmoid最小
def delete_edges_sigmoid(h, g, delete_number):
    delete_list = delete_from_sig_a(h, g, delete_number)
    return delete_list


def delete_from_sig_a(h, g, delete_number):
    # 根据g获取邻接矩阵a
    a = torch.Tensor(nx.adjacency_matrix(g).todense()).type(torch.FloatTensor).cuda()

    # 根据h的sigmoid获取a‘
    sig_a = torch.sigmoid(torch.matmul(h, h.t()))

    # 取出a中为0的位置索引
    one_index = torch.nonzero(a == 1).tolist()
    new_one_index, one_index_pair = change_index(one_index)

    # 根据索引在a’中取出元素
    sig_a_element = sig_a[new_one_index]

    # 两者一起排序
    zip_list = zip(one_index_pair, sig_a_element.tolist())
    temp = sorted(zip_list, key=lambda x: x[1], reverse=False)
    result = zip(*temp)
    new_delete_index, new_delete_element = [x for x in result]

    delete_list = new_delete_index[:delete_number]

    return delete_list


# 在超级节点内部减边
# 失败：准确率太低
def delete_edges_community_nodes_pair_cos(h, g, super_nodes_result_classes):
    res_delete = []
    delete_list = get_edges_super_nodes(g, super_nodes_result_classes)
    cos_distance = get_cos_sim(h)
    delete_cos_distance = cos_distance[change_list(delete_list)]

    print("开始减边....", g)

    for e in delete_list:
        if delete_cos_distance[delete_list.index(e)] <= 0.8:
            res_delete.append(e)

    return res_delete


# 在相似度低的超级节点之间减边
def delete_edges_super_nodes_cos(g, h, result_classes):
    # 减边余弦相似度阈值
    k = 0.35
    k_super_nodes = [0.6, 0.7]

    # 获取节点的余弦相似度矩阵
    cos_sim = get_cos_sim(h)

    # 根据result_classes划分超级节点
    super_nodes_list = result_to_community_order(result_classes, [i for i in range(len(result_classes))])

    # 计算超级节点的嵌入
    super_nodes_embed_list = get_super_nodes_embed(h, super_nodes_list)

    # 计算超级节点嵌入之间的余弦相似度
    # 返回值应该是超过阈值的超级节点对
    delete_super_nodes_pair_list = get_super_nodes_pair_cos_delete(super_nodes_embed_list, k_super_nodes)
    # print("需要减边的超级节点......", delete_super_nodes_pair_list)

    # 在得到的超级节点对之间剪边
    # 每对超级节点对之间的节点对也是按照阈值加边
    delete_list = []
    for i in delete_super_nodes_pair_list:
        delete_edges = delete_edges_between_super_nodes(g, super_nodes_list[i[0]], super_nodes_list[i[1]], k, cos_sim)
        delete_list += delete_edges
    return delete_list


# 超级节点之间减边
# 增加条件：动态设定超级节点阈值k_super_nodes
def delete_edges_super_nodes_cos_dynamic(g, h, result_classes, layer):
    # 减边余弦相似度阈值
    k = 0.4

    # 获取节点的余弦相似度矩阵
    cos_sim = get_cos_sim(h)

    # 根据result_classes划分超级节点
    super_nodes_list = result_to_community_order(result_classes, [i for i in range(len(result_classes))])

    # 计算超级节点的嵌入
    super_nodes_embed_list = get_super_nodes_embed(h, super_nodes_list)

    # 计算超级节点嵌入之间的余弦相似度
    super_nodes_embed_tensor = torch.stack(super_nodes_embed_list)
    cos_super_nodes_sim = get_cos_sim(super_nodes_embed_tensor)

    # 绘制超级节点分布直方图
    # plot_super_nodes_sim(cos_super_nodes_sim)

    # 计算超级节点平均相似度
    k_super_nodes = get_average_sim_super_nodes_2(cos_super_nodes_sim, layer)

    # 返回值超级节点对
    delete_super_nodes_pair_list = get_super_nodes_pair_cos(cos_super_nodes_sim, k_super_nodes)
    # print("需要减边的超级节点......", delete_super_nodes_pair_list)

    # 在得到的超级节点对之间加边
    # 每对超级节点对之间的节点对也是按照阈值加边
    delete_list = []
    for i in delete_super_nodes_pair_list:
        delete_edges = delete_edges_between_super_nodes\
            (g, super_nodes_list[i[0]], super_nodes_list[i[1]], k, cos_sim)
        delete_list += delete_edges
    return delete_list


# 超级节点之间减边
# 增加条件：动态设定超级节点阈值k_super_nodes
# 增加条件：动态设定节点阈值k
def delete_edges_super_nodes_cos_dynamic_2(g, h, result_classes, layer):
    # 减边余弦相似度阈值
    k = 0.5+0.01*layer

    # 获取节点的余弦相似度矩阵
    cos_sim = get_cos_sim(h)

    # 根据result_classes划分超级节点
    super_nodes_list = result_to_community_order(result_classes, [i for i in range(len(result_classes))])

    # 计算超级节点的嵌入
    super_nodes_embed_list = get_super_nodes_embed(h, super_nodes_list)

    # 计算超级节点嵌入之间的余弦相似度
    super_nodes_embed_tensor = torch.stack(super_nodes_embed_list)
    cos_super_nodes_sim = get_cos_sim(super_nodes_embed_tensor)

    # 绘制超级节点分布直方图
    # plot_super_nodes_sim(cos_super_nodes_sim)

    # 计算超级节点平均相似度
    k_super_nodes = get_average_sim_super_nodes(cos_super_nodes_sim)

    # 返回值超级节点对
    delete_super_nodes_pair_list = get_super_nodes_pair_cos(cos_super_nodes_sim, k_super_nodes)
    print("需要减边的超级节点数量......", len(delete_super_nodes_pair_list))

    # 在得到的超级节点对之间加边
    # 每对超级节点对之间的节点对也是按照阈值加边
    delete_list = []
    for i in delete_super_nodes_pair_list:
        delete_edges = delete_edges_between_super_nodes\
            (g, super_nodes_list[i[0]], super_nodes_list[i[1]], k, cos_sim)
        delete_list += delete_edges
    return delete_list


# 超级节点之间减边
# 增加条件：动态设定超级节点阈值k_super_nodes
# 增加条件：动态设定节点阈值k
def delete_edges_super_nodes_cos_dynamic_3(g, h, result_classes, layer, is_change_delete):
    # 减边余弦相似度阈值
    _, k, _, _, _ = read_parameter()
    if is_change_delete:
        k += 0.01
        write_parameter(-1, k, -1, -1, -1)

    # 获取节点的余弦相似度矩阵
    cos_sim = get_cos_sim(h)

    # 根据result_classes划分超级节点
    super_nodes_list = result_to_community_order(result_classes, [i for i in range(len(result_classes))])

    # 计算超级节点的嵌入
    super_nodes_embed_list = get_super_nodes_embed(h, super_nodes_list)

    # 计算超级节点嵌入之间的余弦相似度
    super_nodes_embed_tensor = torch.stack(super_nodes_embed_list)
    cos_super_nodes_sim = get_cos_sim(super_nodes_embed_tensor)

    # 绘制超级节点分布直方图
    # plot_super_nodes_sim(cos_super_nodes_sim)

    # 计算超级节点平均相似度
    k_super_nodes = get_average_sim_super_nodes(cos_super_nodes_sim)

    # 返回值超级节点对
    delete_super_nodes_pair_list = get_super_nodes_pair_cos(cos_super_nodes_sim, k_super_nodes)
    print("需要减边的超级节点数量......", len(delete_super_nodes_pair_list))

    # 在得到的超级节点对之间加边
    # 每对超级节点对之间的节点对也是按照阈值加边
    delete_list = []
    for i in delete_super_nodes_pair_list:
        delete_edges = delete_edges_between_super_nodes\
            (g, super_nodes_list[i[0]], super_nodes_list[i[1]], k, cos_sim)
        delete_list += delete_edges
    return delete_list

# 删除超级节点间所有边
# 尽管会删除很多正常边，但是删除了绝大部分异常边
def delete_edges_super_nodes_all(g, h, result_classes):
    # 减边余弦相似度阈值
    k = 0.98

    # 获取节点的余弦相似度矩阵
    cos_sim = get_cos_sim(h)

    # 根据result_classes划分超级节点
    super_nodes_list = result_to_community_order(result_classes, [i for i in range(len(result_classes))])

    all_edges = []
    interval_edges = []
    delete_list = []

    print("社区内情况........................................")
    for i in range(len(super_nodes_list)):
        ori_sub_g = nx.subgraph(g, super_nodes_list[i])
        all_edges += [e for e in ori_sub_g.edges()]

    for i in g.edges():
        if (i[0], i[1]) in all_edges or (i[1], i[0]) in all_edges:
            continue
        else:
            interval_edges.append(i)

    for e in interval_edges:
        if cos_sim[e[0]][e[1]] <= k:
            delete_list.append(e)

    return delete_list


# 选取混乱超级节点
# 超级节点内部减边
def delete_edges_classes(g, h, result_classes, num, k, label):
    cos_distance = get_cos_sim(h)
    community_list, label_community_list = \
        get_community_from_result_classes(result_classes, [v for v in g.nodes()], label)
    h_community = []
    classes_embed_list = []
    for i in community_list:
        h_community.append(h[i])
        classes_embed_list.append(h[i].mean(dim=0))
    classes_embed_tensor = torch.stack(classes_embed_list)
    cos_classes_sim = get_cos_sim_k(classes_embed_tensor, len(community_list))
    # print(cos_classes_sim)

    s = [nx.average_clustering(nx.Graph(nx.subgraph(g, c))) for c in community_list]

    # s = [len(c) for c in community_list]
    index_list = []
    new_list = s.copy()
    for i in range(1):
        index_list.append(s.index(min(new_list)))
        new_list.remove(min(new_list))
    res = index_list

    print("减边选取的社区：", res)

    all_delete_list = []
    for i in res:
        sub_g = nx.subgraph(g, community_list[i])
        delete_list = [e for e in sub_g.edges()]
        delete_cos_distance = cos_distance[change_list(delete_list)]

        zip_list = zip(delete_list, delete_cos_distance)
        temp = sorted(zip_list, key=lambda x: x[1], reverse=False)
        result = zip(*temp)
        new_delete_list, new_delete_cos_distance = [list(x) for x in result]

        num = int(len(delete_list)*0.5)
        all_delete_list += new_delete_list[:num]

    return all_delete_list



# -----------------------------------------------------------------------------------------------------------------------------------------
# 减边工具方法

# 选取cos指标中小于k的边的数量和剪边准确率
def select_cos_delete_k(new_delete_list, new_delete_euclidean_distance):
    k = 0.3
    delete_list = []
    for i in range(len(new_delete_list)):
        if new_delete_euclidean_distance[i] <= k:
            delete_list.append(new_delete_list[i])
    print("选取cos指标中小于k的边的数量和剪边准确率....")
    return delete_list


# 选取超级节点中的所有边
def get_edges_super_nodes(g, super_nodes_result_classes):
    # 根据result_classes划分超级节点
    super_nodes_list = result_to_community_order\
        (super_nodes_result_classes, [i for i in range(len(super_nodes_result_classes))])
    edges = []
    for com in super_nodes_list:
        sub_g = nx.subgraph(g, com)
        e = sub_g.edges()
        edges += e
    return edges


