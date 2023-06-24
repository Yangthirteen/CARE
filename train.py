import time
import torch
from plot_result import *
from add_delete_fun import *
from model import *
from analysis_community import *
from utils2 import *

upd = 100  # 每隔10轮计算一次acc、记录最大值
# epochs = 3000  # 运行的总轮次
cycles = 2  # 循环次数
add_number = 50000  # 增加边数量
delete_number = 5  # 删除边数量
add_delete_type = 0  # 删除增加模式（0-增加&删除、1-only增加、2-only删除、3-交替（先加后减））
out_dim = 128
test = 0  # (1-绝对正确加边， 0-不做)


def train_h(x, edge_index, g, label, label_k, ori_g, fea_dim):
    h_list = []  # 保存每一层得到的嵌入H
    h = None
    last_max_h = None
    last_max_result_classes = None
    super_result_classes = None

    all_add_list = []
    all_delete_list = []

    global_acc = []  # 准确率变化
    global_train_loss = []  # 损失值
    result_total = []  # 加减边的准确率
    componets = []  # 连通分量数目
    number_of_nodes_all = []  # 连通分量中节点数目
    result_classes = []  # 分类结果

    # 评价指标
    acc_total = []
    nmi_total = []
    ari_total = []
    epoch_total = []
    last_acc_total = []
    last_acc = 0.0

    # 加边统计
    res_list = []

    if test == 1:
        g = add_0(g, label)
        edge_index = get_edge_index(g)

    for i in range(cycles):
        print("加载训练模型...")
        if i < cycles - 1:
            model = GAE(GCNEncoder_scale(fea_dim, out_dim)).cuda()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                model.parameters()), lr=1e-3)
            epochs = 2000
            input = x
        else:
            model = GAE(GCNEncoder_scale(fea_dim, out_dim)).cuda()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                model.parameters()), lr=1e-3)
            epochs = 2000
            input = x

        # 记录每层训练时间细节
        train_time = []

        max_acc = 0.0
        max_nmi = 0.0
        max_ari = 0.0
        max_epoch = 0

        # x = create_new_h(x, g, 0.3, 2)

        print("第" + str(i + 1) + "轮")
        for epoch in range(epochs):
            # print("\n"+"第"+str(i+1)+"轮——EPOCH ###### {} ######".format(epoch))
            model.train()
            model.zero_grad()
            time_begin = time.time()
            h = model.encode(input, edge_index)
            loss = model.recon_loss(h, edge_index)
            loss.backward()
            time_end = time.time()

            optimizer.step()
            # print("loss:", loss.item())
            global_train_loss.append(loss.item())

            # 记录训练时间
            train_time.append(time_end-time_begin)
            # print("model运行时间："+str(time_end-time_begin))

            # 每10轮验证一次acc
            if (epoch+1) % upd == 0:
                result_classes = classify_spectral(h, label_k)
                ac_nmi = nmi(label, result_classes)
                ac_ari = ari(label, result_classes)
                purity = purity_score(label, result_classes)
                global_acc.append(purity)
                if epoch+1 == epochs:
                    last_acc_total.append(purity)
                    last_acc = purity
                if purity > max_acc:
                    max_acc = purity
                    max_nmi = ac_nmi
                    max_ari = ac_ari
                    max_epoch = epoch
                    if i == (cycles - 1):
                        last_max_result_classes = result_classes
                        last_max_h = h

        plot(global_train_loss, "model_gae_h")
        plot(global_acc, "all_acc")
        print("train finished")
        torch.cuda.empty_cache()

        acc_total.append(max_acc)
        nmi_total.append(max_nmi)
        ari_total.append(max_ari)
        epoch_total.append(max_epoch)
        h_list.append(h)

        number_of_nodes = get_number_of_nodes_of_componets(g)
        number_of_nodes_all.append(number_of_nodes)
        componets.append(nx.number_connected_components(g))
        print("连通分量数目：", nx.number_connected_components(g))
        print("连通分量中节点数量：", number_of_nodes)
        print("连通分量数目：", componets)
        print("连通分量节点数目：", number_of_nodes_all)
        print("最大acc:", max_acc, "最好结果轮数：", max_epoch, '最大nmi:', max_nmi, '最大ari：', max_ari)
        print("最后一次的H的acc：", last_acc)

        analysis_super_nodes_big(result_classes, h, g, label)

        # 分析每轮最后得到的聚类结果
        analysis(result_classes, g, label)

        # 改变图结构(最后一轮不改变)
        if i == cycles-1:
            break

        if add_delete_type == 0:
            type_change = 'add+delete'
        elif add_delete_type == 1:
            type_change = 'add'
        elif add_delete_type == 2:
            type_change = 'delete'
        else:
            if i % 2 == 0:
                type_change = 'add'
            else:
                type_change = 'delete'

        # 删减边操作
        g, edge_index, add_list, delete_list, result_txt, res = \
            add_and_delete(h, g, label, type_change, add_number, delete_number, result_classes, ori_g, x,
                           super_result_classes, i)
        print(g)

        all_add_list += add_list
        all_delete_list += delete_list

        result_total.append(result_txt)
        res_list.append(res)

    print_result(result_total)
    print(acc_total, '\n', epoch_total, '\n', nmi_total, '\n', ari_total)
    print("每轮最后的H评价acc：", last_acc_total)
    print("加边统计：", res_list)

    # analysis_add_and_delete(all_add_list, all_delete_list, label)

    return last_max_h, last_max_result_classes, h_list

