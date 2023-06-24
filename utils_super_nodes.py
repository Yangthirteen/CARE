# 超级节点工具类
import matplotlib.pyplot as plt

from utils import *

# 计算超级节点平均相似度
# 获取相似区间
def get_average_sim_super_nodes(cos_super_nodes_sim):
    gap = 0.1
    n = cos_super_nodes_sim.shape[0]
    sum = cos_super_nodes_sim.sum()
    sum -= n
    average = sum/(n*n-n)
    return [average-gap, average+gap]


# 计算超级节点平均相似度
# 获取相似区间
def get_average_sim_super_nodes_2(cos_super_nodes_sim, layer):
    # gap = 0.05+layer*0.005
    gap = 0.05
    n = cos_super_nodes_sim.shape[0]
    sum = cos_super_nodes_sim.sum()
    sum -= n
    average = sum/(n*n-n)
    return [average-gap, average+gap]


# 超级节点阈值写入文件
def write_average(k_super_nodes):
    with open('pic/k_super_nodes.txt', 'a', encoding='utf8') as f:
        f.writelines(str(k_super_nodes)+'\n')


# 程序运行前先清空超级节点阈值文件
def clean_txt():
    with open('pic/k_super_nodes.txt', 'r+') as file:
        file.truncate(0)


# # 分析超级节点之间的异常边所在的节点对的相似度分布
def analysis_err_edges_super_nodes(g, h, result_classes, label):
    # 获取节点的余弦相似度矩阵
    cos_sim = get_cos_sim(h)

    # 根据result_classes划分超级节点
    super_nodes_list = result_to_community_order(result_classes, [i for i in range(len(result_classes))])

    # 计算超级节点的嵌入
    super_nodes_embed_list = get_super_nodes_embed(h, super_nodes_list)

    # 计算超级节点嵌入之间的余弦相似度
    super_nodes_embed_tensor = torch.stack(super_nodes_embed_list)
    cos_super_nodes_sim = get_cos_sim(super_nodes_embed_tensor)

    all_edges = []
    interval_edges = []

    for i in range(len(super_nodes_list)):
        ori_sub_g = nx.subgraph(g, super_nodes_list[i])
        all_edges += [e for e in ori_sub_g.edges()]

    for i in g.edges():
        if (i[0], i[1]) in all_edges or (i[1], i[0]) in all_edges:
            continue
        else:
            interval_edges.append(i)

    err_edges = []
    correct_edges = []
    index = []
    index_c = []
    nei_e = []
    nei_c = []
    for i in interval_edges:
        if label[i[0]] != label[i[1]]:
            index1 = get_index_super_nodes_list(i[0], super_nodes_list)
            index2 = get_index_super_nodes_list(i[1], super_nodes_list)
            index.append((index1, index2))
            nei_e.append(get_num_neighbor(i[0], i[0], g))
            err_edges.append(i)
        else:
            index1 = get_index_super_nodes_list(i[0], super_nodes_list)
            index2 = get_index_super_nodes_list(i[1], super_nodes_list)
            index_c.append((index1, index2))
            nei_c.append(get_num_neighbor(i[0], i[0], g))
            correct_edges.append(i)

    print("错误边数量", len(err_edges))
    print("正确边数量", len(correct_edges))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # numBins = [0,2,4,6,8,10]
    # data = nei_e
    # ax.hist(data, numBins)
    # plt.title('err_edges')
    # plt.savefig('pic/err_edges.png')
    # plt.show()
    # plt.close()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # numBins = [0,2,4,6,8,10]
    # data = nei_c
    # ax.hist(data, numBins)
    # plt.title('err_edges')
    # plt.savefig('pic/err_edges.png')
    # plt.show()
    # plt.close()

    # cos_distance = cos_super_nodes_sim[change_list(index)]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # gap = 0.01
    # num_gap = int(1/gap)
    # numBins = [i/num_gap for i in range(num_gap+1)]
    # data = cos_distance.tolist()
    # ax.hist(data, numBins)
    # plt.title('err_edges')
    # plt.savefig('pic/err_edges.png')
    # plt.show()
    # plt.close()
    #
    # cos_distance = cos_super_nodes_sim[change_list(index_c)]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # gap = 0.01
    # num_gap = int(1/gap)
    # numBins = [i/num_gap for i in range(num_gap+1)]
    # data = cos_distance.tolist()
    # ax.hist(data, numBins)
    # plt.title('correct_edges')
    # plt.savefig('pic/correct_edges.png')
    # plt.show()


# 获取节点在哪个超级节点内
def get_index_super_nodes_list(v, super_nodes_list):
    index = 0
    for i in super_nodes_list:
        if v in i:
            return index
        index += 1
    return -1


# 获取两个节点共同邻居数
def get_num_neighbor(v1, v2, g):
    return len(list(nx.common_neighbors(g, v1, v2)))


# 读取参数（相似度阈值(加边,减边)，超级节点阈值（a, b），聚类数量(num)）
def read_parameter(file_name='pic/parameter.txt'):
    add = None
    delete = None
    top = None
    button = None
    num = None
    file = open(file_name, 'r', encoding='utf-8')
    for line in file.readlines():
        line = line.strip().split(',')
        if len(line) > 0:
            add = float(line[0])
            delete = float(line[1])
            top = float(line[2])
            button = float(line[3])
            num = int(line[4])
    return add, delete, top, button, num


# 写入参数（相似度阈值，超级节点阈值，聚类数量）
def write_parameter(add, delete, top, button, num, file_name='pic/parameter.txt',):
    add_pre, delete_pre, top_pre, button_pre, num_pre = read_parameter(file_name)
    if add == -1:
        add = add_pre
    if delete == -1:
        delete = delete_pre
    if top == -1:
        top = top_pre
    if button == -1:
        button = button_pre
    if num == -1:
        num = num_pre
    with open(file_name, 'w') as file:
        file.write(str(add)+','+str(delete)+','+str(top)+','+str(add)+','+str(button)+','+str(num))
    file.close()
