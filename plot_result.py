import matplotlib.pyplot as plt
import torch
from sklearn import manifold
from matplotlib.backends.backend_pdf import PdfPages


# 绘制loss分布图
def plot(loss_list, result_name):
    epochs_range = range(len(loss_list))
    plt.figure(figsize=(8, 8))
    # 训练损失loss绘制
    plt.subplot(1, 1, 1)
    plt.plot(epochs_range, loss_list, label=result_name)
    # 标识线的位置
    plt.legend(loc='upper right')
    plt.title('Training Loss')

    # 保存图片
    plt.savefig('pic/result_'+result_name+'.png')
    plt.cla()
    plt.close()
    print("plot finished")


# 绘制嵌入分布图
def plot_tsne(X, label, name):
    '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X.cpu().detach().numpy())
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    '''嵌入空间可视化'''

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(label[i]))
    plt.xticks([])
    plt.yticks([])
    with PdfPages('pic/'+name+'.pdf') as pdf:
        pdf.savefig()
    plt.show()
# # 绘制嵌入分布图
# def plot_tsne(X, label):
#     '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
#     tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
#     X_tsne = tsne.fit_transform(X.cpu().detach().numpy())
#     print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
#     '''嵌入空间可视化'''
#
#     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#     X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
#     plt.figure(figsize=(8, 8))
#     for i in range(X_norm.shape[0]):
#         plt.text(X_norm[i, 0], X_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),
#                  fontdict={'weight': 'bold', 'size': 9})
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()


# 绘制超级节点相似度的分布图
def plot_super_nodes_sim(cos_super_nodes):
    cos = cos_super_nodes.detach()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    gap = 0.01
    num_gap = int(1/gap)
    numBins = [i/num_gap for i in range(num_gap+1)]
    data = cos.tolist()
    total_data = []
    for i in data:
        total_data += i
    ax.hist(total_data, numBins)
    plt.title('cos_super_nodes')
    plt.savefig('pic/cos_super_nodes.png')
    plt.show()


# 绘制超级节点相似度的分布图
def plot_classes_sim(cos_classes, name):
    # cos = cos_classes.detach()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    gap = 0.01
    num_gap = int(1/gap)
    numBins = [i/num_gap for i in range(num_gap+1)]
    data = cos_classes.tolist()
    total_data = []
    for i in data:
        total_data += i
    ax.hist(total_data, numBins)
    plt.title(name)
    plt.savefig('pic/'+name+'.png')
    plt.show()
    plt.close()
