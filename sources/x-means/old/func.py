import tensorflow as tf
import numpy as np
from scipy import stats
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans

def xmeans(samples, initial_centroids, kmin, kmax=None, dim=2):
    k = kmin
    cluster_centers = initial_centroids
    while kmax is None or k <= kmax:
        # とりあえずk-means
        model = KMeans(n_clusters=k, init=cluster_centers).fit(samples)

        # centroidsをいれておくはこ
        cluster_centers = np.empty((0,dim), float)
        labels = np.array([])
        for i, centroid in enumerate(model.cluster_centers_):
            new_point1 = np.sort(samples)[0]
            new_point2 = np.sort(samples)[-1]
            new_points = np.array([new_point1, new_point2])

            model_index = (model.labels_ == i)

            # クラスタリングの結果そこになにもクラスタリングされなければさようなら
            if not np.any(model_index):
                continue

            points = np.array(samples[model_index])

            # 2-means
            test_model = KMeans(n_clusters=2, init=new_points).fit(points)

            cluster1 = np.array(points[test_model.labels_ == 0])
            cluster2 = np.array(points[test_model.labels_ == 1])

            bic_parent = bic([points], [centroid])
            bic_child = bic([cluster1, cluster2], test_model.cluster_centers_)
            if bic_child > bic_parent:
                cluster_centers = np.vstack((cluster_centers, test_model.cluster_centers_))
            else:
                cluster_centers = np.vstack((cluster_centers, centroid))
        if k == np.size(cluster_centers, axis=0):
            break
        k = np.size(cluster_centers, axis=0)

    return (model.labels_, cluster_centers)

def log_likelihood(R, dim, clusters, centroids):
    """
    対数尤度関数を返す
    """
    ll = 0
    for k, cluster in enumerate(clusters):
        Rn = np.size(cluster, axis=0)
        t1 = Rn / 2.0 * np.log(2.0 * np.pi)
        variance = cluster_variance(R, clusters, centroids)
        t2 = (Rn * dim) / 2.0 * np.log(variance)
        t3 = (Rn - k) / 2.0
        t4 = Rn * np.log(Rn)
        t5 = Rn * np.log(R)
        ll += t4 - t5 - t1 - t2 - t3
    return ll

def bic(clusters, centroids):
    R = np.sum(np.size(cluster, axis=0) for cluster in clusters)
    dim = clusters[0][0].shape[0]
    pj = np.size(clusters, axis=0) * (dim + 1)

    ll = log_likelihood(R, dim, clusters, centroids)

    # return ll - ((pj / 2) * np.log(R))
    # AIC
    return ll - pj

def cluster_variance(R, clusters, centroids):
    """
    variance
    """
    s = 0
    k = np.size(clusters, axis=0)
    denom = float(R - k)
    for cluster, centroid in zip(clusters, centroids):
        distances = np.linalg.norm(cluster - centroid)
        # データ点と平均の2乗の和
        s += (distances * distances).sum()
    return s / denom

def plot_samples(all_samples, save=False, name='before'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.scatter(all_samples[:, 0], all_samples[:, 1], color='k')
    plt.show()
    if save:
        plt.savefig("img/{}.pdf".format(name))

def plot_clusters(all_samples, labels, n_samples_per_cluster, centroids, edit_centroids=True, save=True, name='after'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
    for i, centroid in enumerate(centroids):
        samples = np.array([data for j, data in enumerate(all_samples) if labels[j]==i])
        plt.scatter(samples[:,0], samples[:,1], c=colour[i])
        if edit_centroids:
            plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
            plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()
    if save:
        plt.savefig("img/{}.pdf".format(name))

def choose_random_centroids(samples, n_clusters, seed=None):
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples), seed=seed)
    begin = [0,]
    size = [n_clusters,]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids
