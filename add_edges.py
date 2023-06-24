from utils2 import *
from utils_super_nodes import *

# 增加边_路径为k策略
# 选取路径长度为k的节点对，按照欧式距离进行排序
# 选择欧式距离小的加边add_number
def add_edges_path_length(h, g, k, add_number):
    add_list = find_path_length_k(g, k)
    euclidean_distance = get_euclidean_distance(h)
    add_euclidean_distance = euclidean_distance[change_list(add_list)]

    zip_list = zip(add_list, add_euclidean_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=False)
    result = zip(*temp)
    new_add_list, new_add_euclidean_distance = [list(x) for x in result]

    add_list = new_add_list[:add_number]

    return add_list


# 增加边_路径为k策略
# 选取路径长度为k的节点对，按照余弦相似度进行排序
# 选择余弦相似度大的加边add_number
def add_edges_path_length(h, g, k, add_number):
    add_list = find_path_length_k(g, k)
    cos_distance = get_cos_sim(h)
    add_cos_distance = cos_distance[change_list(add_list)]

    zip_list = zip(add_list, add_cos_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
    result = zip(*temp)
    new_add_list, new_add_cos_distance = [list(x) for x in result]

    add_list = new_add_list[:add_number]

    return add_list


# 增加边_连通分量策略
# 寻找连通分量之间不存在路径的节点对
# 按照节点对之间的欧式距离大小，选取欧氏距离小的add_number增加边
def add_edges_componets_eu(h, g, add_number):
    add_list = get_all_no_edges_nodes_component(g)
    euclidean_distance = get_euclidean_distance(h)
    add_euclidean_distance = euclidean_distance[change_list(add_list)]

    zip_list = zip(add_list, add_euclidean_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=False)
    result = zip(*temp)
    new_add_list, new_add_euclidean_distance = [list(x) for x in result]

    add_list = new_add_list[:add_number]

    return add_list


# 增加边_连通分量策略
# 寻找连通分量之间不存在路径的节点对
# 按照节点对之间的余弦相似度大小，选取余弦相似度大的增加边add_number
def add_edges_componets_cos(h, g, add_number):
    add_list = get_all_no_edges_nodes_component(g)
    cos_distance = get_cos_sim(h)
    add_cos_distance = cos_distance[change_list(add_list)]

    zip_list = zip(add_list, add_cos_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
    result = zip(*temp)
    new_add_list, new_add_cos_distance = [list(x) for x in result]

    add_list = new_add_list[:add_number]

    return add_list


# 增加边_连通分量策略_重构sigmoid矩阵
# 寻找连通分量之间不存在路径的节点对
def add_edges_componets_sigmoid(h, g, add_number):
    add_list = get_all_no_edges_nodes_component(g)

    # 根据h的sigmoid获取a‘
    sig_a = torch.sigmoid(torch.matmul(h, h.t()))

    add_distance = sig_a[change_list(add_list)]

    zip_list = zip(add_list, add_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
    result = zip(*temp)
    new_add_list, new_add_euclidean_distance = [list(x) for x in result]

    add_list = new_add_list[:add_number]

    return add_list


# 增加边_所有无连边节点对策略
# 寻找图中不存在路径的节点对
# 按照节点对之间的欧式距离大小，选取欧氏距离小的add_number增加边
def add_edges_nodes_pair_eu(h, g, add_number):
    add_list = get_all_no_edges_nodes(g)
    euclidean_distance = get_euclidean_distance(h)
    add_euclidean_distance = euclidean_distance[change_list(add_list)]

    zip_list = zip(add_list, add_euclidean_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=False)
    result = zip(*temp)
    new_add_list, new_add_euclidean_distance = [list(x) for x in result]

    add_list = new_add_list[:add_number]

    return add_list


# 增加边_所有无连边节点对策略
# 寻找图中不存在路径的节点对
# 按照节点对之间的余弦相似度大小，选取余弦相似度大的增加边add_number
def add_edges_nodes_pair_cos(h, g, add_number):
    add_list = get_all_no_edges_nodes(g)
    cos_distance = get_cos_sim(h)
    add_cos_distance = cos_distance[change_list(add_list)]

    zip_list = zip(add_list, add_cos_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
    result = zip(*temp)
    new_add_list, new_add_cos_distance = [list(x) for x in result]

    # add_list = new_add_list[:add_number]

    add_list = select_cos_add_k(new_add_list, new_add_cos_distance)

    return add_list


# 增加边_所有无连边节点对策略_重构sigmoid矩阵
# 寻找图中不存在路径的节点对
def add_edges_nodes_pair_sigmoid(h, g, add_number):
    add_list = add_from_sig_a(h, g, add_number)
    return add_list


def add_from_sig_a(h, g, add_number):
    # 根据g获取邻接矩阵a
    a = torch.Tensor(nx.adjacency_matrix(g).todense()).type(torch.FloatTensor).cuda()

    # 根据h的sigmoid获取a‘
    sig_a = torch.sigmoid(torch.matmul(h, h.t()))

    # 取出a中为0的位置索引
    zero_index = torch.nonzero(a == 0).tolist()
    new_zero_index, zero_index_pair = change_index(zero_index)

    # 根据索引在a’中取出元素
    sig_a_element = sig_a[new_zero_index]

    # 两者一起排序
    zip_list = zip(zero_index_pair, sig_a_element.tolist())
    temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
    result = zip(*temp)
    new_add_index, new_add_element = [x for x in result]

    # 取前k个作为增加的边，查看准确率
    add_list = new_add_index[:add_number]

    return add_list


# 在社区内加边
def add_edges_community_nodes_pair_cos(h, g, result_classes):
    add_list = get_all_no_edges_nodes(g)
    cos_distance = get_cos_sim(h)
    add_cos_distance = cos_distance[change_list(add_list)]

    zip_list = zip(add_list, add_cos_distance)
    temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
    result = zip(*temp)
    new_add_list, new_add_cos_distance = [list(x) for x in result]

    print("开始加边....", g)
    add_list = select_cos_add_node(new_add_list, new_add_cos_distance, result_classes, g)

    return add_list


# 超级节点之间加边
def add_edges_super_nodes_cos(g, h, result_classes, x, label):
    # 加边余弦相似度阈值
    k = 0.997
    k_super_nodes = [0.6, 0.7]

    # 获取节点的余弦相似度矩阵
    cos_sim = get_cos_sim(h)

    # 根据result_classes划分超级节点
    super_nodes_list = result_to_community_order(result_classes, [i for i in range(len(result_classes))])

    # 计算超级节点的嵌入
    super_nodes_embed_list = get_super_nodes_embed(h, super_nodes_list)

    # 计算超级节点嵌入之间的余弦相似度
    super_nodes_embed_tensor = torch.stack(super_nodes_embed_list)
    cos_super_nodes_sim = get_cos_sim(super_nodes_embed_tensor)

    # 分析部分
    # analysis_super_nodes(super_nodes_list, super_nodes_embed_list, g, label, h)
    # analysis_sim_super_nodes(super_nodes_embed_list)

    # 绘制超级节点分布直方图
    plot_super_nodes_sim(cos_super_nodes_sim)

    # 返回值应该是超过阈值的超级节点对
    add_super_nodes_pair_list = get_super_nodes_pair_cos(cos_super_nodes_sim, k_super_nodes)
    print("需要加边的超级节点......", add_super_nodes_pair_list)

    # 在得到的超级节点对之间加边
    # 每对超级节点对之间的节点对也是按照阈值加边
    add_list = []
    for i in add_super_nodes_pair_list:
        add_edges = add_edges_between_super_nodes(g, super_nodes_list[i[0]], super_nodes_list[i[1]], k, cos_sim, add_super_nodes_pair_list)
        if len(add_edges) > 0:
            print(i)
        add_list += add_edges
    return add_list


# 超级节点之间加边
# 增加条件：动态设定超级节点阈值k_super_nodes
def add_edges_super_nodes_cos_dynamic(g, h, result_classes, x, label, layer):
    # 加边余弦相似度阈值
    k = 0.995

    # 获取节点的余弦相似度矩阵
    cos_sim = get_cos_sim(h)

    # 根据result_classes划分超级节点
    super_nodes_list = result_to_community_order(result_classes, [i for i in range(len(result_classes))])

    # 计算超级节点的嵌入
    super_nodes_embed_list = get_super_nodes_embed(h, super_nodes_list)

    # 计算超级节点嵌入之间的余弦相似度
    super_nodes_embed_tensor = torch.stack(super_nodes_embed_list)
    cos_super_nodes_sim = get_cos_sim(super_nodes_embed_tensor)

    analysis_sim_super_nodes(super_nodes_embed_list)

    # 绘制超级节点分布直方图
    plot_super_nodes_sim(cos_super_nodes_sim)

    # 计算超级节点平均相似度
    k_super_nodes = get_average_sim_super_nodes_2(cos_super_nodes_sim, layer)
    write_average(k_super_nodes)

    # 返回值应该是超过阈值的超级节点对
    add_super_nodes_pair_list = get_super_nodes_pair_cos(cos_super_nodes_sim, k_super_nodes)
    # print("需要加边的超级节点......", add_super_nodes_pair_list)

    # 在得到的超级节点对之间加边
    # 每对超级节点对之间的节点对也是按照阈值加边
    add_list = []
    for i in add_super_nodes_pair_list:
        add_edges = add_edges_between_super_nodes\
            (g, super_nodes_list[i[0]], super_nodes_list[i[1]], k, cos_sim)
        # if len(add_edges) > 0:
        #     print(i)
        add_list += add_edges
    return add_list


# 超级节点之间加边
# 增加条件：动态设定超级节点阈值k_super_nodes
# 增加条件：动态设定节点阈值k
def add_edges_super_nodes_cos_dynamic_2(g, h, result_classes, x, label, layer):
    # 加边余弦相似度阈值
    k = 0.998 - 0.001*layer

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
    plot_super_nodes_sim(cos_super_nodes_sim)

    # 计算超级节点平均相似度
    k_super_nodes = get_average_sim_super_nodes(cos_super_nodes_sim)
    write_average(k_super_nodes)

    # 返回值应该是超过阈值的超级节点对
    add_super_nodes_pair_list = get_super_nodes_pair_cos(cos_super_nodes_sim, k_super_nodes)
    print("需要加边的超级节点数量......", len(add_super_nodes_pair_list))

    # 在得到的超级节点对之间加边
    # 每对超级节点对之间的节点对也是按照阈值加边
    add_list = []
    for i in add_super_nodes_pair_list:
        add_edges = add_edges_between_super_nodes\
            (g, super_nodes_list[i[0]], super_nodes_list[i[1]], k, cos_sim)
        # if len(add_edges) > 0:
        #     print(i)
        add_list += add_edges
    return add_list


# 超级节点之间加边
# 增加条件：动态设定超级节点阈值k_super_nodes
# 增加条件：动态设定节点阈值k(无边可加就改变)
def add_edges_super_nodes_cos_dynamic_3(g, h, result_classes, x, label, layer, is_change_add):
    # 加边余弦相似度阈值
    k, _, _, _, _ = read_parameter()
    if is_change_add:
        k -= 0.01
        write_parameter(k, -1, -1, -1, -1)

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
    plot_super_nodes_sim(cos_super_nodes_sim)

    # 计算超级节点平均相似度
    k_super_nodes = get_average_sim_super_nodes(cos_super_nodes_sim)
    write_average(k_super_nodes)

    # 返回值应该是超过阈值的超级节点对
    add_super_nodes_pair_list = get_super_nodes_pair_cos(cos_super_nodes_sim, k_super_nodes)
    print("需要加边的超级节点数量......", len(add_super_nodes_pair_list))

    # 在得到的超级节点对之间加边
    # 每对超级节点对之间的节点对也是按照阈值加边
    add_list = []
    for i in add_super_nodes_pair_list:
        add_edges = add_edges_between_super_nodes\
            (g, super_nodes_list[i[0]], super_nodes_list[i[1]], k, cos_sim)
        # if len(add_edges) > 0:
        #     print(i)
        add_list += add_edges
    return add_list


# 选取纯净超级节点
# 超级节点内部加边
def add_edges_classes(g, h, result_classes, num, k, label):
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

    print("加边选取的社区：", res)

    all_add_list = []

    for i in res:
        sub_g = nx.subgraph(g, community_list[i])
        add_list = get_all_no_edges_nodes(sub_g)
        add_cos_distance = cos_distance[change_list(add_list)]

        zip_list = zip(add_list, add_cos_distance)
        temp = sorted(zip_list, key=lambda x: x[1], reverse=True)
        result = zip(*temp)
        new_delete_list, new_delete_cos_distance = [list(x) for x in result]

        # num = int(len(add_list)*0.001)
        all_add_list += new_delete_list[:num]

    # for i in res:
    #     add_list = []
    #     while len(add_list) < 60:
    #         idx0 = random.randrange(0, len(community_list[i]))
    #         idx1 = random.randrange(0, len(community_list[i]))
    #         if idx0 == idx1 or (community_list[i][idx0], community_list[i][idx1]) in add_list:
    #             continue
    #         add_list.append((community_list[i][idx0], community_list[i][idx1]))
    #     all_add_list += add_list

    return all_add_list


# -----------------------------------------------------------------------------------------------------------------------------------------
# 加边工具方法


# 选取cos指标中大于k的边的数量和加边准确率
def select_cos_add_k(new_add_list, new_add_cos_distance):
    k = 0.99
    add_list = []
    for i in range(len(new_add_list)):
        if new_add_cos_distance[i] >= k:
            add_list.append(new_add_list[i])
    print("选取cos指标中大于k的边的数量和加边准确率....")
    return add_list


# 依据cos在划分的社区中加边，先使用cos计算所有节点对，然后在这些候选里面加到社区上
def select_community_cos_add_k(new_add_list, new_add_cos_distance, result_classes):
    k = 0.99
    add_each_number = 300
    add_list = []
    count_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
    for i in range(len(new_add_list)):
        if new_add_cos_distance[i] >= k and \
                result_classes[new_add_list[i][0]] == result_classes[new_add_list[i][1]] and \
                count_dict[result_classes[new_add_list[i][0]]] < add_each_number:
            add_list.append(new_add_list[i])
            count_dict[result_classes[new_add_list[i][0]]] += 1
    return add_list


# 依据cos在划分的社区中加边，先使用cos计算所有节点对，然后在这些候选里面加到社区上
# 增加条件：不仅在每个社区加边均匀，也要在社区内部加边均匀
def select_community_cos_add_k_in(new_add_list, new_add_cos_distance, result_classes):
    k = 0.995
    add_each_number = 2000
    node_add_edge_limit = 10
    add_list = []
    count_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
    count_dict_node = {}
    for i in range(len(result_classes)):
        count_dict_node[i] = 0
    for i in range(len(new_add_list)):
        if new_add_cos_distance[i] >= k and \
                result_classes[new_add_list[i][0]] == result_classes[new_add_list[i][1]] and \
                count_dict[result_classes[new_add_list[i][0]]] < add_each_number and \
                count_dict_node[new_add_list[i][0]] < node_add_edge_limit and \
                count_dict_node[new_add_list[i][1]] < node_add_edge_limit:
            add_list.append(new_add_list[i])
            count_dict[result_classes[new_add_list[i][0]]] += 1
            count_dict_node[new_add_list[i][0]] += 1
            count_dict_node[new_add_list[i][1]] += 1

    print("划分社区加边数量：", count_dict)
    return add_list


# 依据cos在每个节点上加边，每个节点加边数量有限制
# 增加条件：增加边的两个节点的路径长度要考虑
# 增加条件：度较大节点不加边
def select_cos_add_node(new_add_list, new_add_cos_distance, g):
    top_k = 0.99
    button_k = 0.9
    d_limit = 100
    node_add_edge_limit = 100
    path_limit = 3
    add_list = []
    count_dict_node = {}
    for i in range(len(g.nodes())):
        count_dict_node[i] = 0
    for i in range(len(new_add_list)):
        if top_k >= new_add_cos_distance[i] >= button_k and \
                count_dict_node[new_add_list[i][0]] < node_add_edge_limit and count_dict_node[new_add_list[i][1]] < node_add_edge_limit and \
                ((nx.has_path(g, new_add_list[i][0], new_add_list[i][1]) and nx.shortest_path_length(g, new_add_list[i][0], new_add_list[i][1]) >= path_limit) or nx.has_path(g, new_add_list[i][0], new_add_list[i][1] is False)) and \
                g.degree(new_add_list[i][0]) <= d_limit and g.degree(new_add_list[i][0]) <= d_limit:
            add_list.append(new_add_list[i])
            count_dict_node[new_add_list[i][0]] += 1
            count_dict_node[new_add_list[i][1]] += 1
    return add_list



