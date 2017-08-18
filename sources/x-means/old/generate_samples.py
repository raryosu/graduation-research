# -*- coding: utf-8 -*-

"""
Generate and plot sample data
"""

import tensorflow as tf
import numpy as np
import sys
from sklearn import datasets
from sklearn import metrics
from func import *

DIM = int(sys.argv[1])
K = int(sys.argv[2])
NUM = int(sys.argv[3])

#Generate data
samples, labels = datasets.make_blobs(n_samples=NUM, centers=K, n_features=DIM)
samples = tf.Variable(samples)
labels = tf.Variable(labels)
initial_centroids = choose_random_centroids(samples, 2)
centroids = tf.concat(initial_centroids, 0, name='centroids')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    sample_values = session.run(samples)
    label = session.run(labels)
    centroids = session.run(centroids)

    plot_samples(sample_values, save=True, name='before')

    labels, cluster_centers = xmeans(sample_values, centroids, kmin=2)

    # plot graph
    plot_clusters(sample_values, labels, n_samples_per_cluster, cluster_centers, edit_centroids=True, save=True, name='after')
    plot_clusters(sample_values, labels, n_samples_per_cluster, cluster_centers, edit_centroids=False, save=True, name='after')

    # Calculate criterion
    purity = metrics.adjusted_rand_score(label, labels)
    nmi = metrics.normalized_mutual_info_score(label, labels)
    ari = metrics.adjusted_rand_score(label, labels)

    print("Estimated k = " + str(cluster_centers.size) + ", purity = " + str(purity) + ", NMI = " + str(nmi) + ", ARI = " + str(ari) + "\n")
