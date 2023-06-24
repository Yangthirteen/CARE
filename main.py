from train import *
from classify_fun import *
from utils_super_nodes import *


if __name__ == '__main__':
    begin_time = time.time()

    # 清空超级节点阈值文件
    clean_txt()

    print("加载数据...")
    dataset = 'cora'
    if dataset == 'cora':
        a, g, x, edge_index, label = prepare_data_cora()
        # plot_tsne(x, label, 'raw')
        label_k = 7
        fea_dim = 1433
    elif dataset == 'citeseer':
        a, g, x, edge_index, label = prepare_data_citeseer()
        label_k = 6
        fea_dim = 3703
    elif dataset == 'pubmed':
        a, g, x, edge_index, label = prepare_data_pubmed()
        label_k = 3
        fea_dim = 500
    elif dataset == 'wiki':
        a, g, x, edge_index, label = prepare_data_wiki()
        label_k = 17
        fea_dim = 4973
    else:
        print("数据集错误.....")
        exit(0)

    ori_g = g.copy()

    # 处理全为0的原始节点特征————随机初始化(防止梯度反向传播出现nan)
    for i in range(len(x)):
        if torch.equal(x[i].cpu(), torch.zeros(x[i].shape)):
            x[i] = torch.randn_like(x[i])
            print(x[i])

    h, result_classes, h_list = train_h(x, edge_index, g, label, label_k, ori_g, fea_dim)

    print("生成评价指标结果...")
    ac_nmi = nmi(label, result_classes)
    ac_ari = ari(label, result_classes)
    purity = purity_score(label, result_classes)
    modularity = compute_modularity(ori_g, [i for i in result_classes])
    print("原始图结构：", ori_g)
    print("模块度Q：" + str(modularity))
    print("purity(acc):" + str(purity))
    print("nmi:" + str(ac_nmi))
    print("ari:" + str(ac_ari))

    # compare_add_list_in_structure_hole(result_classes, label, 0, h_list)

    # 分析最后得到的最好结果
    # analysis(result_classes, ori_g, label)

    print("绘制嵌入分布图...")
    plot_tsne(h, label, 'full')
    # plot_tsne(h, result_classes)

    end_time = time.time()
    print("程序运行总时间：", (end_time-begin_time)/60.0, 'min')





