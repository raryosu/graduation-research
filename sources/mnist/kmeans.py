# coding: utf-8

import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans


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

    km = kmeans(X, 10)

    purity = metrics.adjusted_rand_score(TrueLabels, km.labels_)
    nmi = metrics.normalized_mutual_info_score(TrueLabels, km.labels_)
    ari = metrics.adjusted_rand_score(TrueLabels, km.labels_)

    print(str(purity) + ", " + str(nmi) + ", " + str(ari))


if __name__ == '__main__':
    main()
