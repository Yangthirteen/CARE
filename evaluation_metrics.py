from sklearn import metrics
import networkx.algorithms.community as nx_comm
import numpy as np


# NMI标准化互信息
# 问题：某些节点同属一类，分类可能正确，但是标签不对应，可能需要再重新实现
def nmi(truth_class, pre_class):
    nmi = metrics.normalized_mutual_info_score(truth_class, pre_class)
    return nmi


# ARI
def ari(truth_class, pre_class):
    ari = metrics.adjusted_rand_score(truth_class, pre_class)
    return ari


# acc
def purity_score(truth_class, pre_class):
    """A function to compute cluster purity"""
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(truth_class, pre_class)

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# 计算模块度，利用networksX的方法
# 方法：根据global_node_class查找节点类别
# 可控制global_node_class的层数计算不同层的模块度
def compute_modularity(g, result_classes):
    total_list = []
    for i in range(len(result_classes)):
        if result_classes[i] == -1:
            continue
        j = i+1
        class_list = []
        class_list.append(i)
        while j<len(result_classes):
            if result_classes[j]==result_classes[i] and result_classes!=-1:
                class_list.append(j)
                result_classes[j]=-1
            j = j+1
        total_list.append(class_list)
    return nx_comm.modularity(g, total_list)


if __name__ == '__main__':
    t = [0,1,1]
    t2 = [1,1,1]
    p = [1,1,2]

    print(purity_score(p, t))
    print(purity_score(p, t2))



