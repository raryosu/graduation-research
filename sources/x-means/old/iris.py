# -*- coding: utf-8 -*-

"""
Generate and plot sample data
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import metrics
from sklearn.decomposition import PCA
from func import *

n_features = 2
n_clusters = 5
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70
threshold = 0.1e-9

# load iris datasets
iris = datasets.load_iris()
samples = tf.Variable(iris.data[:,:3])
labels = tf.Variable(iris.target)

initial_centroids = choose_random_centroids(samples, 2, seed=seed)
centroids = tf.concat(initial_centroids, 0, name='centroids')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    sample_values = session.run(samples)
    label = session.run(labels)
    centroids = session.run(centroids)

    labels, cluster_centers = xmeans(sample_values, centroids, kmin=2, dim=sample_values.shape[1])

    # Plot graph
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(sample_values[:, 0], sample_values[:, 1], sample_values[:, 2], c=labels,
               cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()

    # Calculate criterion
    purity = metrics.adjusted_rand_score(label, labels)
    nmi = metrics.normalized_mutual_info_score(label, labels)
    ari = metrics.adjusted_rand_score(label, labels)

    print("Estimated k = " + str(cluster_centers.size) + ", purity = " + str(purity) + ", NMI = " + str(nmi) + ", ARI = " + str(ari) + "\n")
