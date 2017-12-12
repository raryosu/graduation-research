# coding: utf-8

import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import logging

logger = logging.getLogger(__name__)
logger.setLevel(10)
sh = logging.StreamHandler()
logger.addHandler(sh)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
sh.setFormatter(formatter)


def prepare_dataset(data_home="./"):
    mnist = fetch_mldata('MNIST original', data_home=data_home)
    X = mnist.data
    Y = mnist.target

    X = X.astype(np.float64)
    X /= X.max()

    return (X, Y)


def kmeans(X, k):
    kmeans = KMeans(n_clusters=k).fit(X)
    return(kmeans)


def main():
    X, TrueLabels = prepare_dataset()

    bandwidth = estimate_bandwidth(X, quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)

    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    purity = metrics.adjusted_rand_score(TrueLabels, labels)
    nmi = metrics.normalized_mutual_info_score(TrueLabels, labels)
    ari = metrics.adjusted_rand_score(TrueLabels, labels)

    print(str(n_clusters_) + ", " + str(purity) + ", " + str(nmi) + ", " + str(ari))


if __name__ == '__main__':
    main()



"""
K       Purity      NMI                 ARI
---------------------------------------------------------------
1       0.0         -6.93889390391e-06  0.0
"""