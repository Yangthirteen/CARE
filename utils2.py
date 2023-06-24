from classify_fun import *
from analysis_result import *


# 分析各种方法得到的簇的纯洁度
# 按照每个簇中占比最多的节点定义纯洁度
# 聚类算法采用谱聚类（之后可替换）
def analysis_super_nodes(h, label, orig):
    print("超级节点结果分析.........")
    super_nodes_number = 300  # 聚类个数
    purity_value = 0.9  # 纯净度阈值

    super_result_classes = classify_spectral(h, super_nodes_number)
    # analysis(super_result_classes, orig, label)

    community_list, label_community_list = \
        get_community_from_result_classes(super_result_classes, [v for v in orig.nodes()], label)

    print("每个簇内情况........................................")
    acc = 0  # 统计超过阈值的簇个数
    cover_nodes = 0
    for i in range(len(community_list)):
        count = Counter(label_community_list[i])
        max_nodes = count.most_common(1)[0][1]
        res = max_nodes / len(label_community_list[i])
        if res >= purity_value:
            acc += 1
            cover_nodes += len(label_community_list[i])
    print("超过阈值的簇个数为：", acc, "簇总个数：", super_nodes_number, "准确率：", acc/super_nodes_number)
    print("覆盖的节点数：", cover_nodes)


def analysis_super_nodes_2(h, label, orig, super_result_classes, super_nodes_number):
    print("纯净度分析.........")
    purity_value = 0.9  # 纯净度阈值

    community_list, label_community_list = \
        get_community_from_result_classes(super_result_classes, [v for v in orig.nodes()], label)

    sum = 0
    acc = 0  # 统计超过阈值的簇个数
    cover_nodes = 0
    for i in range(len(community_list)):
        count = Counter(label_community_list[i])
        max_nodes = count.most_common(1)[0][1]
        res = max_nodes / len(label_community_list[i])
        sum+=res
        if res >= purity_value:
            acc += 1
            cover_nodes += len(label_community_list[i])
    print("超过阈值的簇个数为：", acc, "簇总个数：", str(super_nodes_number), "准确率：", acc/super_nodes_number)
    print("覆盖的节点数：", cover_nodes)
    print(sum/super_nodes_number)


# 输出超级节点之间的相似度
def analysis_sim_super_nodes(super_nodes_embed_list):
    super_nodes_embed_tensor = torch.stack(super_nodes_embed_list)
    cos = get_cos_sim(super_nodes_embed_tensor)
    writetxt2(cos, "cos_super_nodes")


