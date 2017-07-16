# -*- coding: utf-8 -*-

"""
Generate and plot sample data
"""

import tensorflow as tf
import numpy as np
from sklearn import datasets
from functions import *

n_features = 2
n_clusters = 5
n_samples_per_cluster = 1000
seed = np.random.randint(1)
embiggen_factor = 70
threshold = 0.1e-9

samples, labels = datasets.make_blobs(
        n_samples = n_samples_per_cluster,
        centers = n_clusters,
        # random_state=seed
        )
samples = tf.Variable(samples)
labels = tf.Variable(labels)
initial_centroids = choose_random_centroids(samples, n_clusters)
centroids = tf.concat(initial_centroids, 0, name='centroids')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    sample_values = session.run(samples)
    label = session.run(labels)
    centroids = session.run(centroids)
    plot_samples(sample_values, save=True, name='before')
    x_means = XMeans(random_state=1).fit(sample_values)
    plot_clusters(sample_values, x_means.labels_, n_samples_per_cluster, x_means.cluster_centers_, save=True, name='after')
    plot_clusters(sample_values, label, n_samples_per_cluster, centroids, edit_centroids=False, save=True, name='true')
