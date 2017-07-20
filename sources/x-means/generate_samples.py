# -*- coding: utf-8 -*-

"""
Generate and plot sample data
"""

import tensorflow as tf
import numpy as np
from sklearn import datasets
# from functions import *
from func import *

n_features = 2
n_clusters = 5
n_samples_per_cluster = 500
seed = None
embiggen_factor = 70
threshold = 0.1e-9

samples, labels = datasets.make_blobs(
        n_samples = n_samples_per_cluster * n_clusters,
        centers = n_clusters,
        random_state = seed
        )
samples = tf.Variable(samples)
labels = tf.Variable(labels)
initial_centroids = choose_random_centroids(samples, 2, seed=seed)
centroids = tf.concat(initial_centroids, 0, name='centroids')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    sample_values = session.run(samples)
    label = session.run(labels)
    centroids = session.run(centroids)
    plot_samples(sample_values, save=True, name='before')
    # xmeans = XMeans(2).fit(sample_values)
    labels, cluster_centers = xmeans(sample_values, centroids, kmin=2)
    plot_clusters(sample_values, labels, n_samples_per_cluster, cluster_centers, edit_centroids=True, save=True, name='after')
    # plot_clusters(sample_values, xmeans.labels_, n_samples_per_cluster, xmeans.cluster_centers_, edit_centroids=False, save=True, name='after')
    # print(cluster_centers)
    # plot_clusters(sample_values, label, n_samples_per_cluster, centroids, edit_centroids=False, save=True, name='true')
