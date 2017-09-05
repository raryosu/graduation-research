# -*- coding: utf-8 -*-

"""
Generate and plot sample data
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from func import *

n_features = 2
n_clusters = 5
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70
threshold = 0.1e-9

# load wine datasets
wine = np.loadtxt("../datasets/wine.data", delimiter=",")
samples = tf.Variable(wine[:,1:14])
labels = tf.Variable(wine[:, 0])

initial_centroids = choose_random_centroids(samples, 2, seed=seed)
centroids = tf.concat(initial_centroids, 0, name='centroids')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    sample_values = session.run(samples)
    label = session.run(labels)
    centroids = session.run(centroids)
    # plot_samples(sample_values, save=True, name='before')
    labels, cluster_centers = xmeans(sample_values, centroids, kmin=2, dim=sample_values.shape[1])
    # plot_clusters(sample_values, labels, n_samples_per_cluster, cluster_centers, edit_centroids=True, save=True, name='after')
    print(cluster_centers)
    # plot_clusters(sample_values, label, n_samples_per_cluster, centroids, edit_centroids=False, save=True, name='true')
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
