from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from data import *
from evaluation_metrics import *


# kmeans聚类算法
def classify_kmeans(h, k):
    pca = PCA(n_components=8)
    reduced_x = pca.fit_transform(h.cpu().detach().numpy())
    reduced_x = torch.Tensor(reduced_x)

    zmax = reduced_x.max(dim=1, keepdim=True)[0]
    zmin = reduced_x.min(dim=1, keepdim=True)[0]
    z_std = (reduced_x - zmin) / (zmax - zmin)

    # 指定要聚多少个类别，以及拟合数据X。
    kmeans_label = KMeans(init='k-means++', n_clusters=k, random_state=0).fit(z_std.detach().numpy())

    return kmeans_label.labels_.tolist()


# 谱聚类算法
def classify_spectral(h, k):
    m = np.matmul(h.cpu().detach().numpy(), np.transpose(h.cpu().detach().numpy()))
    Cluster = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=0).fit_predict(m)
    return Cluster.tolist()


# 验证聚类代码正确性
if __name__ == '__main__':
    a, g, x, edge_index, label = prepare_data_cora()
    result_classes = classify_spectral(x, 7)

    print("评价指标结果：")
    nmi = nmi(label, result_classes)
    ari = ari(label, result_classes)
    purity = purity_score(label, result_classes)
    # modularity = compute_modularity(g, [i for i in result_classes])
    #
    # print("模块度Q：" + str(modularity))
    print("purity(acc):" + str(purity))
    print("nmi:" + str(nmi))
    print("ari:" + str(ari))


