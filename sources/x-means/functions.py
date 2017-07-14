# -*- coding: utf-8 -*-

"""
x-means algorithm

Hagihara Ryosuke
2017.07.04

FIXME:
    k-meansのClass化(現在はsklearnを利用)
"""

import tensorflow as tf
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

def plot_samples(all_samples, save=False, name='before'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.scatter(all_samples[:, 0], all_samples[:, 1], color='k')
    plt.show()
    if save:
        plt.savefig("img/{}.pdf".format(name))

def plot_clusters(all_samples, labels, centroids, n_samples_per_cluster, save=True, name='after'):
    """
    Plot the clusters with different colour for each cluster
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
        samples = np.array([data for j, data in enumerate(all_samples) if labels[j]==i])
        plt.scatter(samples[:,0], samples[:,1], c=colour[i])
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()
    if save:
        plt.savefig("img/{}.pdf".format(name))

def choose_random_centroids(samples, n_clusters, seed=None):
    """
    Initialisation
    - Select `n_clusters` numboer of random points
    """
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples), seed=seed)
    begin = [0,]
    size = [n_clusters,]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids

def assign_to_nearest(samples, centroids):
    """
    Finds the nearest centroid for each samples
    - Compute distances from each sample to centroids
    """
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum( tf.square(
        tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)

    nearest_indices = mins
    return nearest_indices

def update_centroids(samples, nearest_indices, n_clusters):
    """
    Updates the centroid to be the mean of all samples associated with it.
    - Separate samples by "nearest_indices"
    - Compute the mean of each cluster and set centroid there
    """
    nearest_indices = tf.to_int32(nearest_indices)
    # Clustering the samples
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    # New centroids position is mean the samples
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    return new_centroids

def update(samples, centroids, n_clusters):
    nearest_indices = assign_to_nearest(samples, centroids)
    centroids = update_centroids(samples, nearest_indices, n_clusters)
    return (nearest_indices, centroids)

def k_means(samples, centroids, n_clusters, threshold):
    """
    k-meansをしてくれる
    """
    while True:
        old_centroids = centroids
        nearest_indices, centroids = update(samples, old_centroids, n_clusters)
        diff = (old_centroids - centroids)
        if len(np.where(np.array( diff.eval().flatten() <= threshold ) )[0]) == n_clusters*2:
            break
    return (samples, nearest_indices, centroids)

class XMeans:
    """
    x-means法を行うクラス
    """

    def __init__(self, k_init = 2, **k_means_args):
        """
        k_init : The initial number of clusters applied to KMeans()
        """
        self.k_init = k_init
        self.k_means_args = k_means_args

    def fit(self, X):
        """
        x-means法を使ってデータXをクラスタリングする
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        self.__clusters = [] 

        clusters = self.Cluster.build(X, KMeans(self.k_init, **self.k_means_args).fit(X))
        self.__recursively_split(clusters)

        self.labels_ = np.empty(X.shape[0], dtype = np.intp)
        for i, c in enumerate(self.__clusters):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])

        return self

    def __recursively_split(self, clusters):
        """
        引数のclustersを再帰的に分割する
        clusters : list-like object, which contains instances of 'XMeans.Cluster'
        """
        for cluster in clusters:
            if cluster.size <= 3:
                self.__clusters.append(cluster)
                continue

            k_means = KMeans(2, **self.k_means_args).fit(cluster.data)
            c1, c2 = self.Cluster.build(cluster.data, k_means, cluster.index)

            # return log_likelihood - (n_clusters/2) * math.log(sample.size)
            # bic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df * np.log(cluster.size)
            bic = (c1.log_likelihood() + c2.log_likelihood()) + (cluster.size / 2) * np.log(cluster.size)

            if bic > cluster.bic():
                self.__recursively_split([c1, c2])
            else:
                self.__clusters.append(cluster)

    class Cluster:
        """
        k-means法によって生成されたクラスタに関する情報を持ち、尤度やBICの計算を行うクラス
        """

        @classmethod
        def build(cls, X, k_means, index = None):
            if np.any(index == None):
                index = np.array(range(0, X.shape[0]))
            labels = range(0, k_means.get_params()["n_clusters"])

            return tuple(cls(X, index, k_means, label) for label in labels)

        # index: Xの各行におけるサンプルが元データの何行目のものかを示すベクトル
        def __init__(self, X, index, k_means, label):
            self.data = X[k_means.labels_ == label]
            self.index = index[k_means.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            self.center = k_means.cluster_centers_[label]
            self.cov = np.cov(self.data.T)

        def log_likelihood(self):
            return sum(stats.multivariate_normal.logpdf(x, self.center, self.cov) for x in self.data)

        def bic(self):
            return -2 * self.log_likelihood() + self.df * np.log(self.size)

def x_means(samples, threshold, seed=None):
    """
    x-meansをしてくれる
    """
    n_clusters = 2
    __clusters = []
    __centroids = []

    centroids = choose_random_centroids(samples, n_clusters, seed)
    samples, nearest_indices, centroids = k_means(samples, centroids, n_clusters, threshold)
    c0, c1 = split_clusters(samples, nearest_indices.eval())
    print(c0)

    x_means(c0, threshold, seed)
    x_means(c1, threshold, seed)

def split_clusters(samples, nearest_indices):
    c0 = np.array([samples[x] for x, data in enumerate(nearest_indices) if data == 0])
    c1 = np.array([samples[x] for x, data in enumerate(nearest_indices) if data == 1])
    return (c0, c1)

def log_likelihood(data, n_clusters):
    import math
    likelihood = np.array([])
    for x in range(n_clusters):
        dim = len(data)
        cov = 1/(data.size - n_clusters) * (np.linalg.norm(data - np.expand_dims(np.mean(data, axis=1), axis=1)))
        tmp =  data[x].size/2 * (math.log(2 * math.pi) - dim * math.log(cov) - (data[x].size - n_clusters)/data[x].size + math.log(data[x].size/data.size))
        np.append(likelihood, tmp)
    return likelihood

def bic(log_likelihood, sample, n_clusters):
    import math
    return log_likelihood - (n_clusters/2) * math.log(sample.size)

